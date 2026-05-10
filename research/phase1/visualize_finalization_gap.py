import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "research" / "phase1" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "research" / "phase1" / ".cache"))
if str(REPO_ROOT / "research" / "phase1") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "research" / "phase1"))

from analysis_utils import active_position_records, load_traces, self_final_step  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize finalization-gap dynamics from trace JSONL.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.trace_path)

    gaps = []
    correct_by_gap = {}
    for sample in traces:
        prompt_len = sample.get("prompt_length", len(sample.get("prompt_ids", [])))
        target_ids = sample.get("target_ids", [])
        per_pos = active_position_records(sample)
        for global_pos, records in per_pos.items():
            if global_pos < prompt_len:
                continue
            target_idx = global_pos - prompt_len
            first_commit = None
            for rec in records:
                if rec["committed"]:
                    first_commit = rec["global_iter"]
                    break
            self_final = self_final_step(records)
            if first_commit is None or self_final is None:
                continue
            gap = first_commit - self_final
            gaps.append(gap)
            final_top1 = records[-1]["topk_ids"][0]
            correct = None
            if 0 <= target_idx < len(target_ids):
                correct = int(final_top1 == target_ids[target_idx])
            bucket = str(gap)
            correct_by_gap.setdefault(bucket, {"correct": 0, "total": 0})
            if correct is not None:
                correct_by_gap[bucket]["correct"] += correct
                correct_by_gap[bucket]["total"] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(gaps, bins=min(30, max(5, len(set(gaps)) or 5)), color="#4063d8", alpha=0.85)
    axes[0].set_title("Finalization Gap Histogram")
    axes[0].set_xlabel("t_commit - t_self_final")
    axes[0].set_ylabel("Token count")

    sorted_items = sorted((int(k), v) for k, v in correct_by_gap.items())
    xs = [item[0] for item in sorted_items]
    ys = [None if item[1]["total"] == 0 else item[1]["correct"] / item[1]["total"] for item in sorted_items]
    axes[1].plot(xs, ys, marker="o", color="#d64b4b")
    axes[1].set_title("Correctness by Finalization Gap")
    axes[1].set_xlabel("t_commit - t_self_final")
    axes[1].set_ylabel("Correct ratio")
    axes[1].set_ylim(0, 1.05)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output_dir / "finalization_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "num_traces": len(traces),
        "num_gap_points": len(gaps),
        "mean_gap": None if not gaps else sum(gaps) / len(gaps),
        "gap_gt3_ratio": None if not gaps else sum(1 for gap in gaps if gap > 3) / len(gaps),
        "correctness_by_gap": {
            key: {
                "correct": value["correct"],
                "total": value["total"],
                "ratio": None if value["total"] == 0 else value["correct"] / value["total"],
            }
            for key, value in correct_by_gap.items()
        },
    }
    with (args.output_dir / "finalization_gap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
