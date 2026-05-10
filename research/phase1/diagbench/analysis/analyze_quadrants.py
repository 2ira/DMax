import argparse
import csv
import json
from pathlib import Path

from .analysis_utils import active_position_records, load_traces


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze confidence/consistency quadrants from phase-1 traces.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--consistency-k", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.trace_path)

    rows = []
    quadrant_stats = {
        "high_c_high_s": {"count": 0, "correct": 0},
        "high_c_low_s": {"count": 0, "correct": 0},
        "low_c_high_s": {"count": 0, "correct": 0},
        "low_c_low_s": {"count": 0, "correct": 0},
    }

    for sample in traces:
        prompt_len = sample.get("prompt_length", len(sample.get("prompt_ids", [])))
        target_ids = sample.get("target_ids", [])
        per_pos = active_position_records(sample)
        for global_pos, records in per_pos.items():
            if global_pos < prompt_len:
                continue
            target_idx = global_pos - prompt_len
            final_correct = None
            if 0 <= target_idx < len(target_ids):
                final_correct = int(records[-1]["topk_ids"][0] == target_ids[target_idx])
            for idx, rec in enumerate(records):
                if idx + 1 < args.consistency_k:
                    consistency = 0
                else:
                    window = [records[j]["topk_ids"][0] for j in range(idx - args.consistency_k + 1, idx + 1)]
                    consistency = int(len(set(window)) == 1)
                confidence = rec["topk_probs"][0]
                high_c = confidence >= args.confidence_threshold
                key = (
                    "high_c_high_s" if high_c and consistency else
                    "high_c_low_s" if high_c else
                    "low_c_high_s" if consistency else
                    "low_c_low_s"
                )
                quadrant_stats[key]["count"] += 1
                if final_correct is not None:
                    quadrant_stats[key]["correct"] += final_correct
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "task": sample.get("task"),
                        "position": global_pos,
                        "global_iter": rec["global_iter"],
                        "confidence": confidence,
                        "consistency": consistency,
                        "quadrant": key,
                        "final_correct": final_correct,
                    }
                )

    with (args.output_dir / "quadrants.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for key, value in quadrant_stats.items():
        summary[key] = {
            "count": value["count"],
            "correct_ratio": None if value["count"] == 0 else value["correct"] / value["count"],
        }
    with (args.output_dir / "quadrants_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
