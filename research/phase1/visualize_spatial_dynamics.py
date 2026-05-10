import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "research" / "phase1" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "research" / "phase1" / ".cache"))
if str(REPO_ROOT / "research" / "phase1") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "research" / "phase1"))

from analysis_utils import load_traces  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize step-position dynamics from trace JSONL.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--fields",
        nargs="*",
        default=["committed", "changed", "extra.eligible", "extra.within_frontier_window", "extra.frozen"],
    )
    return parser.parse_args()


def _pick_sample(traces, sample_id, sample_index):
    if sample_id is not None:
        for sample in traces:
            if sample.get("sample_id") == sample_id:
                return sample
        raise ValueError(f"sample_id not found: {sample_id}")
    if sample_index < 0 or sample_index >= len(traces):
        raise IndexError(f"sample_index out of range: {sample_index}")
    return traces[sample_index]


def _extract_field(step, local_idx, field_name):
    if field_name.startswith("extra."):
        key = field_name.split(".", 1)[1]
        values = step.get("extra", {}).get(key)
        if values is None:
            return np.nan
        return values[local_idx]
    values = step.get(field_name)
    if values is None:
        return np.nan
    return values[local_idx]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.trace_path)
    sample = _pick_sample(traces, args.sample_id, args.sample_index)

    prompt_len = sample.get("prompt_length", len(sample.get("prompt_ids", [])))
    seq_len = len(sample.get("final_output_ids", []))
    steps = sample.get("steps", [])
    if seq_len == 0:
        seq_len = prompt_len + sample.get("target_length", 0)

    target_positions = list(range(prompt_len, seq_len))
    if not target_positions:
        raise ValueError("No target positions found for spatial visualization.")

    num_steps = len(steps)
    matrices = {field: np.full((num_steps, len(target_positions)), np.nan, dtype=float) for field in args.fields}

    pos_to_col = {pos: idx for idx, pos in enumerate(target_positions)}
    for step_idx, step in enumerate(steps):
        positions = step.get("positions", [])
        for local_idx, global_pos in enumerate(positions):
            if global_pos not in pos_to_col:
                continue
            col = pos_to_col[global_pos]
            for field in args.fields:
                value = _extract_field(step, local_idx, field)
                matrices[field][step_idx, col] = value

    fig, axes = plt.subplots(len(args.fields), 1, figsize=(12, 2.8 * len(args.fields)), sharex=True)
    if len(args.fields) == 1:
        axes = [axes]
    for ax, field in zip(axes, args.fields):
        im = ax.imshow(matrices[field], aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_ylabel("step")
        ax.set_title(field)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    axes[-1].set_xlabel("target token position")
    fig.suptitle(f"Spatial dynamics: {sample.get('sample_id')}")
    fig.tight_layout()
    fig.savefig(args.output_dir / "spatial_dynamics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / "spatial_dynamics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_id": sample.get("sample_id"),
                "prompt_length": prompt_len,
                "num_steps": num_steps,
                "fields": args.fields,
                "target_positions": target_positions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
