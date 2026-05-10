import math

import torch
import torch.nn.functional as F

from dinfer.decoding.parallel_strategy import (
    ThresholdParallelDecoder,
    _get_prob_stats,
    broadcast_if_needed,
    get_transfer_index_uniform,
)


def _record_uniform_trace(decoder, *, logits, x_before, x_after, active_index, mask_index, high_conf_index, changed_mask, block_start, block_end, break_flag):
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
    )


def _build_next_embeddings(decoder, logits, x, block_start, block_end, active_index, embedding_layer, prev_embeddings=None, top_k=1):
    curr_x = x[:, block_start:block_end]
    new_mask_index = (curr_x == decoder.mask_id)
    soft_cond = active_index & (~new_mask_index)
    if prev_embeddings is None:
        base_embeds = embedding_layer(curr_x)
    else:
        base_embeds = prev_embeddings
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


class MinimalJotDecoder(ThresholdParallelDecoder):
    """Approximate per-token early stopping for block-diffusion decoding.

    This is a research baseline, not an official reproduction of Jot.
    """

    def __init__(self, stability_steps=3, stability_prob=0.9, **kwargs):
        super().__init__(**kwargs)
        self.stability_steps = stability_steps
        self.stability_prob = stability_prob
        self._prev_top1 = {}
        self._streak = {}
        self._frozen = {}

    def block_init(self, block_x, block_id):
        self._prev_top1.clear()
        self._streak.clear()
        self._frozen.clear()

    def decode_uniform(self, logits, block_start, block_end, x, active_index, embedding_layer, prev_embeddings=None, iter_threshold=None, top_k=1):
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        curr_x = x[:, block_start:block_end]
        trace_x_before = curr_x.clone() if self.trace_recorder is not None else None

        x0, high_conf_index, max_probs, max_indices = get_transfer_index_uniform(
            logits,
            self.temperature,
            mask_index,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )

        key = (block_start, block_end)
        prev_top1 = self._prev_top1.get(key)
        streak = self._streak.get(key)
        frozen = self._frozen.get(key)
        if prev_top1 is None:
            prev_top1 = max_indices.clone()
            streak = torch.ones_like(max_indices, dtype=torch.int64)
            frozen = torch.zeros_like(active_index, dtype=torch.bool)
        else:
            same = max_indices == prev_top1
            streak = torch.where(same, streak + 1, torch.ones_like(streak))
            prev_top1 = max_indices.clone()

        newly_frozen = active_index & (~mask_index) & (streak >= self.stability_steps) & (max_probs >= self.stability_prob)
        frozen = frozen | newly_frozen

        if mask_index.any():
            cond = active_index & (~mask_index)
            update_mask = (high_conf_index | cond) & (~frozen)
        else:
            update_mask = active_index & (~frozen)

        changed_mask = update_mask & (x0 != curr_x)
        if update_mask.any():
            curr_x[update_mask] = x0[update_mask]

        self._prev_top1[key] = prev_top1
        self._streak[key] = streak
        self._frozen[key] = frozen

        effective_active = active_index & (~frozen)
        if effective_active.any() and (max_probs[effective_active] >= 0.9).all():
            break_flag = True
        elif not changed_mask.any() and not mask_index[effective_active].any():
            break_flag = True
        else:
            break_flag = False

        _record_uniform_trace(
            self,
            logits=logits,
            x_before=trace_x_before,
            x_after=curr_x,
            active_index=active_index,
            mask_index=mask_index,
            high_conf_index=high_conf_index,
            changed_mask=changed_mask,
            block_start=block_start,
            block_end=block_end,
            break_flag=break_flag,
        )

        if break_flag:
            return break_flag, prev_embeddings
        next_embeddings = _build_next_embeddings(
            self,
            logits,
            x,
            block_start,
            block_end,
            effective_active,
            embedding_layer,
            prev_embeddings=prev_embeddings,
            top_k=top_k,
        )
        return break_flag, next_embeddings


class MinimalSTDecDecoder(ThresholdParallelDecoder):
    """Approximate spatio-temporal threshold adaptation.

    This is a research baseline, not an official reproduction of STDec.
    """

    def __init__(self, consistency_steps=3, temporal_bonus=0.08, spatial_bonus=0.05, **kwargs):
        super().__init__(**kwargs)
        self.consistency_steps = consistency_steps
        self.temporal_bonus = temporal_bonus
        self.spatial_bonus = spatial_bonus
        self._prev_top1 = {}
        self._streak = {}

    def block_init(self, block_x, block_id):
        self._prev_top1.clear()
        self._streak.clear()

    def decode_uniform(self, logits, block_start, block_end, x, active_index, embedding_layer, prev_embeddings=None, iter_threshold=None, top_k=1):
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        curr_x = x[:, block_start:block_end]
        trace_x_before = curr_x.clone() if self.trace_recorder is not None else None

        if self.temperature == 0:
            x0 = torch.argmax(logits, dim=-1)
        else:
            logits_with_noise = logits
            x0 = torch.argmax(logits_with_noise, dim=-1)
        x0_p, max_probs, max_indices = _get_prob_stats(
            logits,
            x0,
            use_float64=self.use_float64,
            x_is_argmax=math.isclose(self.temperature, 0.0),
        )
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, float("-inf")))

        key = (block_start, block_end)
        prev_top1 = self._prev_top1.get(key)
        streak = self._streak.get(key)
        if prev_top1 is None:
            prev_top1 = max_indices.clone()
            streak = torch.ones_like(max_indices, dtype=torch.int64)
        else:
            same = max_indices == prev_top1
            streak = torch.where(same, streak + 1, torch.ones_like(streak))
            prev_top1 = max_indices.clone()
        self._prev_top1[key] = prev_top1
        self._streak[key] = streak

        temporal_score = torch.where(streak >= self.consistency_steps, torch.ones_like(confidence), torch.zeros_like(confidence))
        left_visible = torch.zeros_like(mask_index, dtype=torch.float32)
        right_visible = torch.zeros_like(mask_index, dtype=torch.float32)
        if mask_index.shape[1] > 1:
            left_visible[:, 1:] = (~mask_index[:, :-1]).to(torch.float32)
            right_visible[:, :-1] = (~mask_index[:, 1:]).to(torch.float32)
        spatial_score = torch.clamp(left_visible + right_visible, max=1.0)

        adjusted_conf = confidence + self.temporal_bonus * temporal_score + self.spatial_bonus * spatial_score
        is_low_conf = mask_index & (adjusted_conf < iter_threshold)
        has_failure = torch.cumsum(is_low_conf.long(), dim=1) > 0
        candidates = mask_index & (~has_failure)
        batch_has_selection = candidates.any(dim=-1, keepdim=True)
        mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
        first_mask_token = (mask_cumsum == 1) & mask_index
        high_conf_index = torch.where(batch_has_selection, candidates, first_mask_token)

        if mask_index.any():
            cond = active_index & (~mask_index)
            update_mask = high_conf_index | cond
        else:
            update_mask = active_index

        changed_mask = update_mask & (x0 != curr_x)
        if update_mask.any():
            curr_x[update_mask] = x0[update_mask]
        if self._should_sync_across_ranks():
            broadcast_if_needed(x.data)

        if (max_probs[active_index] >= 0.9).all():
            break_flag = True
        elif not changed_mask.any():
            break_flag = True
        else:
            break_flag = False

        _record_uniform_trace(
            self,
            logits=logits,
            x_before=trace_x_before,
            x_after=curr_x,
            active_index=active_index,
            mask_index=mask_index,
            high_conf_index=high_conf_index,
            changed_mask=changed_mask,
            block_start=block_start,
            block_end=block_end,
            break_flag=break_flag,
        )

        if break_flag:
            return break_flag, prev_embeddings
        next_embeddings = _build_next_embeddings(
            self,
            logits,
            x,
            block_start,
            block_end,
            active_index,
            embedding_layer,
            prev_embeddings=prev_embeddings,
            top_k=top_k,
        )
        return break_flag, next_embeddings
