import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .analysis_utils import active_position_records, load_traces


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MoE router stability from phase-1 traces.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.trace_path)

    layer_stats = defaultdict(lambda: {"stable": 0, "total": 0})
    for sample in traces:
        per_pos = active_position_records(sample)
        for records in per_pos.values():
            prev_router = None
            for rec in records:
                router = rec.get("router_topk")
                if not router:
                    continue
                if prev_router is not None:
                    for layer_idx, layer_topk in enumerate(router):
                        same = int(layer_topk == prev_router[layer_idx])
                        layer_stats[layer_idx]["stable"] += same
                        layer_stats[layer_idx]["total"] += 1
                prev_router = router

    rows = []
    for layer_idx in sorted(layer_stats):
        info = layer_stats[layer_idx]
        rows.append(
            {
                "layer_idx": layer_idx,
                "num_pairs": info["total"],
                "stability": None if info["total"] == 0 else info["stable"] / info["total"],
            }
        )

    with (args.output_dir / "router_stability.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    with (args.output_dir / "router_stability_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"num_layers": len(rows)}, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
