import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .. import PHASE1_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PHASE1_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PHASE1_ROOT / ".cache"))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize suite-level score/speed tradeoffs.")
    parser.add_argument("--suite-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(args.suite_results)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)

    summary = {}
    for experiment, exp_rows in grouped.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for row in exp_rows:
            score = _to_float(row.get("Score"))
            tpf = _to_float(row.get("TPF"))
            tps = _to_float(row.get("TPS"))
            label = row["method"]
            if score is not None and tpf is not None:
                axes[0].scatter(tpf, score, label=label)
                axes[0].annotate(label, (tpf, score), fontsize=8)
            if score is not None and tps is not None:
                axes[1].scatter(tps, score, label=label)
                axes[1].annotate(label, (tps, score), fontsize=8)
        axes[0].set_xlabel("TPF")
        axes[0].set_ylabel("Score")
        axes[0].set_title(f"{experiment}: Score vs TPF")
        axes[1].set_xlabel("TPS")
        axes[1].set_ylabel("Score")
        axes[1].set_title(f"{experiment}: Score vs TPS")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{experiment}_pareto.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        summary[experiment] = [
            {
                "method": row["method"],
                "Score": _to_float(row.get("Score")),
                "TPF": _to_float(row.get("TPF")),
                "TPS": _to_float(row.get("TPS")),
                "NFE": _to_float(row.get("NFE")),
                "premature_finalization_rate": _to_float(row.get("premature_finalization_rate")),
            }
            for row in exp_rows
        ]

    with (args.output_dir / "suite_visualization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
