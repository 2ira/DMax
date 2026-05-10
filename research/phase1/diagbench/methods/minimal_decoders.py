import math

import torch
import torch.nn.functional as F

from dinfer.decoding.parallel_strategy import (
    CreditThresholdParallelDecoder,
    _get_prob_stats,
    broadcast_if_needed,
)


def _record_uniform_trace(
    decoder,
    *,
    logits,
    x_before,
    x_after,
    active_index,
    mask_index,
    high_conf_index,
    changed_mask,
    block_start,
    block_end,
    break_flag,
    extra_fields=None,
):
    if decoder.trace_recorder is None:
        return
    context = decoder._trace_context or {}
    decoder.trace_recorder.record_uniform_step(
        logits=logits,
        x_before=x_before,
        x_after=x_after,
        active_index=active_index,
        mask_index=mask_index,
        high_conf_index=high_conf_index,
        changed_mask=changed_mask,
        block_start=context.get("block_start", block_start),
        block_end=context.get("block_end", block_end),
        block_id=context.get("block_id", -1),
        block_step=context.get("block_step", -1),
        global_iter=context.get("global_iter", -1),
        is_cross_block=context.get("is_cross_block", False),
        elapsed_ms=decoder._trace_elapsed_ms,
        break_flag=break_flag,
        router_topk=decoder._trace_router_topk,
        extra_fields=extra_fields,
    )


def _build_next_embeddings(
    decoder,
    logits,
    x,
    block_start,
    block_end,
    active_index,
    embedding_layer,
    prev_embeddings=None,
    top_k=1,
):
    curr_x = x[:, block_start:block_end]
    new_mask_index = curr_x == decoder.mask_id
    soft_cond = active_index & (~new_mask_index)
    if prev_embeddings is None:
        base_embeds = embedding_layer(curr_x)
    else:
        base_embeds = prev_embeddings
        hard_refresh_mask = (~soft_cond) & (curr_x != decoder.mask_id)
        if hard_refresh_mask.any():
            base_embeds[hard_refresh_mask] = embedding_layer(curr_x[hard_refresh_mask])
    if not soft_cond.any():
        if decoder._should_sync_across_ranks():
            broadcast_if_needed(base_embeds.data)
        return base_embeds

    if top_k == 1:
        max_indices = torch.argmax(logits, dim=-1)
        topk_indices = max_indices.unsqueeze(-1)
        topk_probs = _get_prob_stats(
            logits,
            max_indices,
            use_float64=decoder.use_float64,
            x_is_argmax=True,
        )[0].unsqueeze(-1)
    else:
        probs = F.softmax(logits.to(torch.float64 if decoder.use_float64 else torch.float32), dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)

    residual_probs = torch.clamp(1.0 - topk_probs.sum(dim=-1, keepdim=True), min=0.0)
    topk_embeds = embedding_layer(topk_indices)
    mask_embed, mask_norm = decoder._get_mask_embed_and_norm(embedding_layer, x.device)
    topk_weighted = (topk_embeds * topk_probs.unsqueeze(-1)).sum(dim=2)
    mask_weighted = mask_embed * residual_probs
    soft_embeds = topk_weighted + mask_weighted

    current_norm = torch.norm(soft_embeds, p=2, dim=-1, keepdim=True)
    topk_norms = torch.norm(topk_embeds, p=2, dim=-1)
    expected_topk_norm = (topk_norms * topk_probs).sum(dim=-1, keepdim=True)
    expected_mask_norm = mask_norm * residual_probs
    target_norm = expected_topk_norm + expected_mask_norm
    soft_embeds = soft_embeds * (target_norm / (current_norm + 1e-6))
    if soft_embeds.dtype != base_embeds.dtype:
        soft_embeds = soft_embeds.to(base_embeds.dtype)
    base_embeds[soft_cond] = soft_embeds[soft_cond]
    if decoder._should_sync_across_ranks():
        broadcast_if_needed(base_embeds.data)
    return base_embeds


def _update_temporal_state(prev_top1, streak, max_indices):
    if prev_top1 is None or streak is None:
        prev_top1 = max_indices.clone()
        streak = torch.ones_like(max_indices, dtype=torch.int64)
    else:
        same = max_indices == prev_top1
        streak = torch.where(same, streak + 1, torch.ones_like(streak))
        prev_top1 = max_indices.clone()
    return prev_top1, streak


def _normalized_negative_entropy(logits, entropy):
    vocab_size = max(int(logits.shape[-1]), 2)
    max_entropy = math.log(vocab_size)
    return 1.0 - torch.clamp(entropy / max_entropy, min=0.0, max=1.0)


def _prefix_select(mask_index, eligible):
    is_failure = mask_index & (~eligible)
    has_failure = torch.cumsum(is_failure.long(), dim=1) > 0
    candidates = mask_index & (~has_failure) & eligible
    batch_has_selection = candidates.any(dim=-1, keepdim=True)
    mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
    first_mask_token = (mask_cumsum == 1) & mask_index
    return torch.where(batch_has_selection, candidates, first_mask_token)


def _frontier_window_mask(mask_index, window_size):
    if window_size is None:
        return torch.ones_like(mask_index, dtype=torch.bool)
    if window_size <= 0:
        return torch.zeros_like(mask_index, dtype=torch.bool)
    seq_len = mask_index.shape[1]
    device = mask_index.device
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(mask_index)
    first_mask_idx = torch.argmax(mask_index.long(), dim=1, keepdim=True)
    has_mask = mask_index.any(dim=1, keepdim=True)
    max_allowed = first_mask_idx + (window_size - 1)
    allowed = positions <= max_allowed
    return allowed & has_mask


def _select_candidates(mask_index, eligible, selection_mode, frontier_window=None):
    constrained = eligible
    within_window = None
    if frontier_window is not None:
        within_window = _frontier_window_mask(mask_index, frontier_window)
        constrained = constrained & within_window
    if selection_mode == "prefix":
        return _prefix_select(mask_index, constrained), within_window
    if selection_mode == "arbitrary":
        batch_has_selection = constrained.any(dim=-1, keepdim=True)
        mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
        first_mask_token = (mask_cumsum == 1) & mask_index
        return torch.where(batch_has_selection, constrained, first_mask_token), within_window
    raise ValueError(f"Unsupported selection_mode: {selection_mode}")


class ConfigurableRuleDecoder(CreditThresholdParallelDecoder):
    """Configurable training-free decoder family for controlled comparisons.

    This is a research framework, not an official reproduction of all compared
    papers. It provides a shared implementation surface so methods differ only
    in their decision rule, while runtime controls remain fixed.
    """

    def __init__(
        self,
        *,
        score_mode="confidence",
        score_threshold=0.5,
        stability_steps=0,
        require_stability=False,
        stability_only=False,
        credit_enabled=False,
        temporal_bonus=0.0,
        spatial_bonus=0.0,
        early_stop_mode="default",
        prophet_patience=3,
        prophet_score_threshold=0.9,
        selection_mode="prefix",
        frontier_window=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.score_mode = score_mode
        self.score_threshold = score_threshold
        self.stability_steps = stability_steps
        self.require_stability = require_stability
        self.stability_only = stability_only
        self.credit_enabled = credit_enabled
        self.temporal_bonus = temporal_bonus
        self.spatial_bonus = spatial_bonus
        self.early_stop_mode = early_stop_mode
        self.prophet_patience = prophet_patience
        self.prophet_score_threshold = prophet_score_threshold
        self.selection_mode = selection_mode
        self.frontier_window = frontier_window
        self._prev_top1 = {}
        self._streak = {}
        self._frozen = {}

    def block_init(self, block_x, block_id):
        self._prev_top1.clear()
        self._streak.clear()
        self._frozen.clear()
        self._credit_mats.clear()
        self._credit_iters.clear()

    def _score_from_mode(self, logits, mask_index, max_probs, topk_probs, entropy):
        if self.score_mode == "confidence":
            score = max_probs
        elif self.score_mode == "margin":
            if topk_probs.shape[-1] >= 2:
                score = topk_probs[..., 0] - topk_probs[..., 1]
            else:
                score = topk_probs[..., 0]
        elif self.score_mode == "negative_entropy":
            score = _normalized_negative_entropy(logits, entropy)
        elif self.score_mode == "none":
            score = torch.zeros_like(max_probs)
        elif self.score_mode == "random_uniform":
            score = torch.rand_like(max_probs)
        else:
            raise ValueError(f"Unsupported score_mode: {self.score_mode}")
        return torch.where(mask_index, score, torch.full_like(score, float("-inf")))

    def _stdec_adjustment(self, mask_index, score, streak):
        if self.temporal_bonus == 0.0 and self.spatial_bonus == 0.0:
            return score
        temporal_score = torch.where(
            streak >= max(self.stability_steps, 1),
            torch.ones_like(score),
            torch.zeros_like(score),
        )
        left_visible = torch.zeros_like(mask_index, dtype=score.dtype)
        right_visible = torch.zeros_like(mask_index, dtype=score.dtype)
        if mask_index.shape[1] > 1:
            left_visible[:, 1:] = (~mask_index[:, :-1]).to(score.dtype)
            right_visible[:, :-1] = (~mask_index[:, 1:]).to(score.dtype)
        spatial_score = torch.clamp(left_visible + right_visible, max=1.0)
        return score + self.temporal_bonus * temporal_score + self.spatial_bonus * spatial_score

    def decode_uniform(
        self,
        logits,
        block_start,
        block_end,
        x,
        active_index,
        embedding_layer,
        prev_embeddings=None,
        iter_threshold=None,
        top_k=1,
    ):
        mask_index = x[:, block_start:block_end] == self.mask_id
        curr_x = x[:, block_start:block_end]
        trace_x_before = curr_x.clone() if self.trace_recorder is not None else None
        if iter_threshold is None:
            iter_threshold = self.score_threshold

        key = (block_start, block_end)
        used_logits = self._apply_credit_fusion(logits, mask_index, key) if self.credit_enabled else logits

        x0 = torch.argmax(used_logits, dim=-1)
        x0_p, max_probs, max_indices = _get_prob_stats(
            used_logits,
            x0,
            use_float64=self.use_float64,
            x_is_argmax=math.isclose(self.temperature, 0.0),
        )
        probs = F.softmax(used_logits.to(torch.float64 if self.use_float64 else torch.float32), dim=-1)
        topk_width = min(2, probs.shape[-1])
        topk_probs, _ = torch.topk(probs, topk_width, dim=-1)
        entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)

        prev_top1, streak = _update_temporal_state(
            self._prev_top1.get(key),
            self._streak.get(key),
            max_indices,
        )
        self._prev_top1[key] = prev_top1
        self._streak[key] = streak

        score = self._score_from_mode(used_logits, mask_index, x0_p, topk_probs, entropy)
        adjusted_score = self._stdec_adjustment(mask_index, score, streak)

        if self.stability_only:
            eligible = mask_index & (streak >= max(self.stability_steps, 1))
        elif self.require_stability:
            eligible = mask_index & (adjusted_score >= iter_threshold) & (streak >= max(self.stability_steps, 1))
        else:
            eligible = mask_index & (adjusted_score >= iter_threshold)

        high_conf_index, within_window = _select_candidates(
            mask_index,
            eligible,
            selection_mode=self.selection_mode,
            frontier_window=self.frontier_window,
        )

        frozen = self._frozen.get(key)
        if frozen is None:
            frozen = torch.zeros_like(active_index, dtype=torch.bool)

        if self.early_stop_mode == "jot":
            freeze_score = score if self.score_mode != "none" else x0_p
            newly_frozen = (
                active_index
                & (~mask_index)
                & (streak >= max(self.stability_steps, 1))
                & (freeze_score >= max(self.prophet_score_threshold, iter_threshold))
            )
            frozen = frozen | newly_frozen
            self._frozen[key] = frozen
            effective_active = active_index & (~frozen)
        else:
            effective_active = active_index

        if mask_index.any():
            cond = active_index & (~mask_index)
            update_mask = (high_conf_index | cond) & (~frozen)
        else:
            update_mask = active_index & (~frozen)

        changed_mask = update_mask & (x0 != curr_x)
        if update_mask.any():
            curr_x[update_mask] = x0[update_mask]
        if self._should_sync_across_ranks():
            broadcast_if_needed(x.data)

        if effective_active.any() and (x0_p[effective_active] >= 0.9).all():
            break_flag = True
        elif self.early_stop_mode == "prophet" and effective_active.any():
            prophet_ready = (
                (streak[effective_active] >= self.prophet_patience).all()
                and (score[effective_active] >= self.prophet_score_threshold).all()
            )
            break_flag = bool(prophet_ready)
        elif not changed_mask.any():
            break_flag = True
        else:
            break_flag = False

        _record_uniform_trace(
            self,
            logits=used_logits,
            x_before=trace_x_before,
            x_after=curr_x,
            active_index=active_index,
            mask_index=mask_index,
            high_conf_index=high_conf_index,
            changed_mask=changed_mask,
            block_start=block_start,
            block_end=block_end,
            break_flag=break_flag,
            extra_fields={
                "score": score,
                "adjusted_score": adjusted_score,
                "streak": streak,
                "eligible": eligible,
                "frozen": frozen,
                "within_frontier_window": within_window if within_window is not None else torch.ones_like(mask_index, dtype=torch.bool),
            },
        )

        if break_flag:
            if not (x.data == self.mask_id).any():
                self._credit_mats.clear()
                self._credit_iters.clear()
            return break_flag, prev_embeddings

        next_embeddings = _build_next_embeddings(
            self,
            used_logits,
            x,
            block_start,
            block_end,
            effective_active,
            embedding_layer,
            prev_embeddings=prev_embeddings,
            top_k=top_k,
        )
        if not (x.data == self.mask_id).any():
            self._credit_mats.clear()
            self._credit_iters.clear()
        return break_flag, next_embeddings


class MinimalJotDecoder(ConfigurableRuleDecoder):
    def __init__(self, stability_steps=3, stability_prob=0.9, **kwargs):
        credit_enabled = kwargs.pop("credit_enabled", False)
        super().__init__(
            score_mode="confidence",
            score_threshold=stability_prob,
            stability_steps=stability_steps,
            require_stability=False,
            stability_only=False,
            credit_enabled=credit_enabled,
            early_stop_mode="jot",
            prophet_score_threshold=stability_prob,
            **kwargs,
        )


class MinimalSTDecDecoder(ConfigurableRuleDecoder):
    def __init__(self, consistency_steps=3, temporal_bonus=0.08, spatial_bonus=0.05, **kwargs):
        threshold = kwargs.pop("threshold")
        credit_enabled = kwargs.pop("credit_enabled", False)
        super().__init__(
            score_mode="confidence",
            score_threshold=threshold,
            stability_steps=consistency_steps,
            require_stability=False,
            stability_only=False,
            credit_enabled=credit_enabled,
            temporal_bonus=temporal_bonus,
            spatial_bonus=spatial_bonus,
            threshold=threshold,
            **kwargs,
        )


class MinimalProphetDecoder(ConfigurableRuleDecoder):
    def __init__(self, prophet_patience=3, prophet_score_threshold=0.9, **kwargs):
        credit_enabled = kwargs.pop("credit_enabled", False)
        threshold = kwargs.get("threshold", 0.5)
        super().__init__(
            score_mode="confidence",
            score_threshold=threshold,
            credit_enabled=credit_enabled,
            early_stop_mode="prophet",
            prophet_patience=prophet_patience,
            prophet_score_threshold=prophet_score_threshold,
            **kwargs,
        )


def build_decoder_from_method(method_name, method_spec, *, mask_id, eos_id):
    family = method_spec.family
    params = dict(method_spec.params)
    common = {
        "temperature": 0,
        "mask_id": mask_id,
        "eos_id": eos_id,
    }

    if family == "standard":
        threshold = params.pop("threshold", 0.5)
        return ConfigurableRuleDecoder(
            score_mode="confidence",
            score_threshold=threshold,
            threshold=threshold,
            **common,
        )
    if family == "rule":
        threshold = params.pop("threshold", 0.5)
        return ConfigurableRuleDecoder(
            score_threshold=threshold,
            threshold=threshold,
            **params,
            **common,
        )
    if family == "stdec":
        threshold = params.pop("threshold", 0.5)
        return MinimalSTDecDecoder(
            threshold=threshold,
            **params,
            **common,
        )
    if family == "jot":
        threshold = params.pop("threshold", 0.5)
        return MinimalJotDecoder(
            threshold=threshold,
            **params,
            **common,
        )
    if family == "prophet":
        threshold = params.pop("threshold", 0.5)
        return MinimalProphetDecoder(
            threshold=threshold,
            **params,
            **common,
        )
    raise ValueError(f"Unsupported method family for {method_name}: {family}")
