import json
from pathlib import Path
from typing import Any

from ..analysis.analysis_utils import active_position_records, load_traces, self_final_step


def compute_premature_finalization_metrics(trace_path: str | Path) -> dict[str, Any]:
    traces = load_traces(trace_path)
    committed_positions = 0
    premature_positions = 0
    per_sample = []

    for sample in traces:
        sample_committed = 0
        sample_premature = 0
        per_pos = active_position_records(sample)
        for global_pos, records in per_pos.items():
            first_commit_idx = None
            first_commit_token = None
            for idx, rec in enumerate(records):
                if rec["committed"]:
                    first_commit_idx = idx
                    first_commit_token = rec["state_token"]
                    break
            if first_commit_idx is None:
                continue

            committed_positions += 1
            sample_committed += 1
            changed_after_commit = any(
                later["state_token"] != first_commit_token for later in records[first_commit_idx + 1 :]
            )
            if changed_after_commit:
                premature_positions += 1
                sample_premature += 1

        per_sample.append(
            {
                "sample_id": sample["sample_id"],
                "committed_positions": sample_committed,
                "premature_positions": sample_premature,
                "premature_rate": None if sample_committed == 0 else sample_premature / sample_committed,
            }
        )

    return {
        "num_samples": len(traces),
        "committed_positions": committed_positions,
        "premature_positions": premature_positions,
        "premature_finalization_rate": None if committed_positions == 0 else premature_positions / committed_positions,
        "per_sample": per_sample,
    }


def compute_self_finalization_gap_metrics(trace_path: str | Path) -> dict[str, Any]:
    traces = load_traces(trace_path)
    gaps = []
    for sample in traces:
        per_pos = active_position_records(sample)
        for _, records in per_pos.items():
            self_final = self_final_step(records)
            first_commit = None
            for rec in records:
                if rec["committed"]:
                    first_commit = rec["global_iter"]
                    break
            if self_final is not None and first_commit is not None:
                gaps.append(first_commit - self_final)
    return {
        "num_positions": len(gaps),
        "mean_self_finalization_gap": None if not gaps else sum(gaps) / len(gaps),
        "gap_gt3_ratio": None if not gaps else sum(1 for gap in gaps if gap > 3) / len(gaps),
    }


def write_trace_metrics(trace_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "premature": compute_premature_finalization_metrics(trace_path),
        "self_finalization": compute_self_finalization_gap_metrics(trace_path),
    }
    with (output_dir / "trace_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return payload
