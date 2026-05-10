import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .. import PHASE1_ROOT
from ..analysis.analysis_utils import load_traces

os.environ.setdefault("MPLCONFIGDIR", str(PHASE1_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PHASE1_ROOT / ".cache"))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize aggregated early-stop dynamics from trace JSONL.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.trace_path)

    active_counts = defaultdict(list)
    committed_counts = defaultdict(list)
    changed_counts = defaultdict(list)
    frozen_counts = defaultdict(list)
    break_counts = defaultdict(list)

    for sample in traces:
        for step in sample.get("steps", []):
            t = step["global_iter"]
            active_counts[t].append(len(step.get("positions", [])))
            committed_counts[t].append(sum(step.get("committed", [])))
            changed_counts[t].append(sum(step.get("changed", [])))
            frozen_counts[t].append(sum(step.get("extra", {}).get("frozen", [])))
            break_counts[t].append(int(step.get("break_flag", False)))

    xs = sorted(active_counts.keys())
    active_mean = [sum(active_counts[t]) / len(active_counts[t]) for t in xs]
    committed_mean = [sum(committed_counts[t]) / len(committed_counts[t]) for t in xs]
    changed_mean = [sum(changed_counts[t]) / len(changed_counts[t]) for t in xs]
    frozen_mean = [sum(frozen_counts[t]) / len(frozen_counts[t]) for t in xs]
    break_ratio = [sum(break_counts[t]) / len(break_counts[t]) for t in xs]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(xs, active_mean, label="active_positions")
    axes[0].plot(xs, committed_mean, label="committed_positions")
    axes[0].plot(xs, changed_mean, label="changed_positions")
    axes[0].plot(xs, frozen_mean, label="frozen_positions")
    axes[0].set_ylabel("mean count")
    axes[0].legend()
    axes[0].set_title("Per-step token dynamics")

    axes[1].plot(xs, break_ratio, color="#c44e52", label="break_ratio")
    axes[1].set_xlabel("global refinement step")
    axes[1].set_ylabel("ratio")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].set_title("Early-stop trigger ratio")

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output_dir / "early_stop_dynamics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / "early_stop_dynamics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "steps": xs,
                "active_mean": active_mean,
                "committed_mean": committed_mean,
                "changed_mean": changed_mean,
                "frozen_mean": frozen_mean,
                "break_ratio": break_ratio,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
