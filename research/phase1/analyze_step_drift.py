import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from analysis_utils import active_position_records, approximate_jsd, load_traces


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze per-step distribution drift from phase-1 traces.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.trace_path)

    bucket = defaultdict(lambda: {"jsd_sum": 0.0, "count": 0, "entropy_sum": 0.0, "margin_sum": 0.0})
    for sample in traces:
        per_pos = active_position_records(sample)
        for records in per_pos.values():
            for prev, curr in zip(records, records[1:]):
                jsd = approximate_jsd(prev["topk_ids"], prev["topk_probs"], curr["topk_ids"], curr["topk_probs"])
                step_bucket = bucket[curr["global_iter"]]
                step_bucket["jsd_sum"] += jsd
                step_bucket["entropy_sum"] += curr["entropy"]
                step_bucket["margin_sum"] += curr["margin"]
                step_bucket["count"] += 1

    rows = []
    for step in sorted(bucket):
        info = bucket[step]
        rows.append(
            {
                "global_iter": step,
                "count": info["count"],
                "mean_jsd": info["jsd_sum"] / info["count"] if info["count"] else None,
                "mean_entropy": info["entropy_sum"] / info["count"] if info["count"] else None,
                "mean_margin": info["margin_sum"] / info["count"] if info["count"] else None,
            }
        )

    with (args.output_dir / "step_drift.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "num_samples": len(traces),
        "num_steps": len(rows),
        "mean_jsd_overall": None if not rows else sum(row["mean_jsd"] for row in rows if row["mean_jsd"] is not None) / len(rows),
    }
    with (args.output_dir / "step_drift_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
