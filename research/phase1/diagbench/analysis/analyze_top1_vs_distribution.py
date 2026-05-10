import argparse
import csv
import json
from pathlib import Path

from .analysis_utils import active_position_records, approximate_jsd, load_traces


def parse_args():
    parser = argparse.ArgumentParser(description="Compare top-1 stability against distribution stability.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jsd-threshold", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.trace_path)

    rows = []
    disagree = 0
    total = 0
    for sample in traces:
        per_pos = active_position_records(sample)
        for global_pos, records in per_pos.items():
            for prev, curr in zip(records, records[1:]):
                jsd = approximate_jsd(prev["topk_ids"], prev["topk_probs"], curr["topk_ids"], curr["topk_probs"])
                top1_stable = int(prev["topk_ids"][0] == curr["topk_ids"][0])
                dist_stable = int(jsd <= args.jsd_threshold)
                total += 1
                if top1_stable != dist_stable:
                    disagree += 1
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "position": global_pos,
                        "prev_iter": prev["global_iter"],
                        "curr_iter": curr["global_iter"],
                        "top1_stable": top1_stable,
                        "dist_stable": dist_stable,
                        "jsd": jsd,
                    }
                )

    with (args.output_dir / "top1_vs_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "num_pairs": total,
        "disagreement_ratio": None if total == 0 else disagree / total,
        "jsd_threshold": args.jsd_threshold,
    }
    with (args.output_dir / "top1_vs_distribution_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
