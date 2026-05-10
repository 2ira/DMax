import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_traces(path: str | Path) -> list[dict[str, Any]]:
    traces = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def active_position_records(sample: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_pos = defaultdict(list)
    for step in sample.get("steps", []):
        positions = step.get("positions", [])
        for idx, global_pos in enumerate(positions):
            by_pos[global_pos].append(
                {
                    "global_iter": step["global_iter"],
                    "block_id": step["block_id"],
                    "block_step": step["block_step"],
                    "masked": bool(step["masked"][idx]),
                    "committed": bool(step["committed"][idx]),
                    "changed": bool(step["changed"][idx]),
                    "state_token": step["state_tokens"][idx],
                    "prev_state_token": step["prev_state_tokens"][idx],
                    "topk_ids": step["topk_ids"][idx],
                    "topk_probs": step["topk_probs"][idx],
                    "margin": step["margin"][idx],
                    "entropy": step["entropy"][idx],
                    "router_topk": None if step.get("router_topk") is None else [layer[idx] for layer in step["router_topk"]],
                }
            )
    return by_pos


def first_true_step(records: list[dict[str, Any]], key: str) -> int | None:
    for rec in records:
        if rec[key]:
            return rec["global_iter"]
    return None


def self_final_step(records: list[dict[str, Any]]) -> int | None:
    top1_seq = [rec["topk_ids"][0] for rec in records]
    for idx, token_id in enumerate(top1_seq):
        if all(token_id == later for later in top1_seq[idx:]):
            return records[idx]["global_iter"]
    return None


def online_stability_step(records: list[dict[str, Any]], k: int = 3) -> int | None:
    top1_seq = [rec["topk_ids"][0] for rec in records]
    for idx in range(k - 1, len(top1_seq)):
        window = top1_seq[idx - k + 1 : idx + 1]
        if len(set(window)) == 1:
            return records[idx]["global_iter"]
    return None


def approximate_jsd(topk_ids_a, topk_probs_a, topk_ids_b, topk_probs_b) -> float:
    dist_a = {tok: prob for tok, prob in zip(topk_ids_a, topk_probs_a)}
    dist_b = {tok: prob for tok, prob in zip(topk_ids_b, topk_probs_b)}
    union = sorted(set(dist_a) | set(dist_b))
    pa = [dist_a.get(tok, 0.0) for tok in union]
    pb = [dist_b.get(tok, 0.0) for tok in union]
    pa_other = max(0.0, 1.0 - sum(pa))
    pb_other = max(0.0, 1.0 - sum(pb))
    pa.append(pa_other)
    pb.append(pb_other)
    m = [(a + b) / 2.0 for a, b in zip(pa, pb)]

    def _kl(p, q):
        total = 0.0
        for p_i, q_i in zip(p, q):
            if p_i > 0:
                total += p_i * math.log(max(p_i, 1e-12) / max(q_i, 1e-12))
        return total

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
