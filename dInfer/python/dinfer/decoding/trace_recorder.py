import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F


class TraceRecorder:
    """Collect per-sample generation traces and persist them as JSONL.

    The recorder keeps one in-memory object per sample in the current batch and
    appends step-level diagnostics during decoding. Callers are expected to:

    1. `start_batch(sample_infos)` before `dllm.generate(...)`
    2. Let the decoder call `record_uniform_step(...)` on every refinement step
    3. `finish_batch(final_outputs)` after generation finishes
    """

    def __init__(
        self,
        output_path: str | Path,
        topk: int = 8,
        record_router: bool = False,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.topk = topk
        self.record_router = record_router
        self._fh = self.output_path.open("a", encoding="utf-8")
        self._active_samples: list[dict[str, Any]] = []

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def start_batch(self, sample_infos: list[dict[str, Any]]) -> None:
        self._active_samples = []
        for info in sample_infos:
            payload = dict(info)
            payload.setdefault("steps", [])
            self._active_samples.append(payload)

    def finish_batch(self, final_outputs: torch.Tensor) -> None:
        if not self._active_samples:
            return
        outputs_cpu = final_outputs.detach().cpu().tolist()
        for idx, sample in enumerate(self._active_samples):
            sample["final_output_ids"] = outputs_cpu[idx]
            self._fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._active_samples = []

    def _serialize_router(
        self,
        router_topk: Optional[torch.Tensor],
        batch_idx: int,
        local_positions: torch.Tensor,
    ) -> Optional[list[list[list[int]]]]:
        if router_topk is None:
            return None
        # router_topk: [num_layers, batch, seq_len, top_k]
        router_slice = router_topk[:, batch_idx, local_positions].detach().cpu()
        return [
            layer_positions.tolist()
            for layer_positions in router_slice
        ]

    def record_uniform_step(
        self,
        *,
        logits: torch.Tensor,
        x_before: torch.Tensor,
        x_after: torch.Tensor,
        active_index: torch.Tensor,
        mask_index: torch.Tensor,
        high_conf_index: torch.Tensor,
        changed_mask: torch.Tensor,
        block_start: int,
        block_end: int,
        block_id: int,
        block_step: int,
        global_iter: int,
        is_cross_block: bool,
        elapsed_ms: float,
        break_flag: bool,
        router_topk: Optional[torch.Tensor] = None,
        extra_fields: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        if not self._active_samples:
            return

        probs = F.softmax(logits.to(torch.float32), dim=-1)
        entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)
        topk = min(self.topk, probs.shape[-1])
        topk_probs, topk_ids = torch.topk(probs, k=topk, dim=-1)
        if topk > 1:
            margin = topk_probs[..., 0] - topk_probs[..., 1]
        else:
            margin = topk_probs[..., 0]

        for batch_idx, sample in enumerate(self._active_samples):
            local_positions = torch.nonzero(active_index[batch_idx], as_tuple=False).flatten()
            if local_positions.numel() == 0:
                step_payload = {
                    "global_iter": int(global_iter),
                    "block_id": int(block_id),
                    "block_step": int(block_step),
                    "block_start": int(block_start),
                    "block_end": int(block_end),
                    "is_cross_block": bool(is_cross_block),
                    "elapsed_ms": float(elapsed_ms),
                    "break_flag": bool(break_flag),
                    "positions": [],
                    "masked": [],
                    "committed": [],
                    "changed": [],
                    "state_tokens": [],
                    "prev_state_tokens": [],
                    "topk_ids": [],
                    "topk_probs": [],
                    "margin": [],
                    "entropy": [],
                    "router_topk": None,
                    "extra": {},
                }
                sample["steps"].append(step_payload)
                continue

            global_positions = (local_positions + block_start).detach().cpu().tolist()
            step_payload = {
                "global_iter": int(global_iter),
                "block_id": int(block_id),
                "block_step": int(block_step),
                "block_start": int(block_start),
                "block_end": int(block_end),
                "is_cross_block": bool(is_cross_block),
                "elapsed_ms": float(elapsed_ms),
                "break_flag": bool(break_flag),
                "positions": global_positions,
                "masked": mask_index[batch_idx, local_positions].detach().cpu().to(torch.int32).tolist(),
                "committed": high_conf_index[batch_idx, local_positions].detach().cpu().to(torch.int32).tolist(),
                "changed": changed_mask[batch_idx, local_positions].detach().cpu().to(torch.int32).tolist(),
                "state_tokens": x_after[batch_idx, local_positions].detach().cpu().tolist(),
                "prev_state_tokens": x_before[batch_idx, local_positions].detach().cpu().tolist(),
                "topk_ids": topk_ids[batch_idx, local_positions].detach().cpu().tolist(),
                "topk_probs": topk_probs[batch_idx, local_positions].detach().cpu().tolist(),
                "margin": margin[batch_idx, local_positions].detach().cpu().tolist(),
                "entropy": entropy[batch_idx, local_positions].detach().cpu().tolist(),
                "router_topk": self._serialize_router(router_topk, batch_idx, local_positions),
                "extra": {},
            }
            if extra_fields:
                for key, value in extra_fields.items():
                    if value is None:
                        continue
                    sliced = value[batch_idx, local_positions]
                    if torch.is_tensor(sliced):
                        sliced_cpu = sliced.detach().cpu()
                        if sliced_cpu.dtype == torch.bool:
                            step_payload["extra"][key] = sliced_cpu.to(torch.int32).tolist()
                        else:
                            step_payload["extra"][key] = sliced_cpu.tolist()
            sample["steps"].append(step_payload)
