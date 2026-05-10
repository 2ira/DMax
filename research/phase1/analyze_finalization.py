import argparse
import csv
import json
from pathlib import Path

from analysis_utils import active_position_records, first_true_step, load_traces, online_stability_step, self_final_step


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze token finalization dynamics from phase-1 traces.")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stability-k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = load_traces(args.trace_path)

    rows = []
    gaps = []
    correctness_rows = 0
    correctness_hits = 0
    gap_gt3 = 0

    for sample in traces:
        prompt_len = sample.get("prompt_length", len(sample.get("prompt_ids", [])))
        target_ids = sample.get("target_ids", [])
        per_pos = active_position_records(sample)
        for global_pos, records in per_pos.items():
            if global_pos < prompt_len:
                continue
            target_idx = global_pos - prompt_len
            t_first = records[0]["global_iter"] if records else None
            t_self_final = self_final_step(records)
            t_stable = online_stability_step(records, args.stability_k)
            t_commit = first_true_step(records, "committed")
            final_top1 = records[-1]["topk_ids"][0]
            correct = None
            if 0 <= target_idx < len(target_ids):
                correct = int(final_top1 == target_ids[target_idx])
                correctness_rows += 1
                correctness_hits += correct
            gap = None
            if t_commit is not None and t_self_final is not None:
                gap = t_commit - t_self_final
                gaps.append(gap)
                if gap > 3:
                    gap_gt3 += 1
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "task": sample.get("task"),
                    "position": global_pos,
                    "target_index": target_idx,
                    "t_first": t_first,
                    "t_stable_k": t_stable,
                    "t_self_final": t_self_final,
                    "t_commit": t_commit,
                    "gap_commit_minus_self_final": gap,
                    "final_top1": final_top1,
                    "correct": correct,
                }
            )

    csv_path = args.output_dir / "finalization_tokens.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "num_samples": len(traces),
        "num_token_rows": len(rows),
        "mean_gap": None if not gaps else sum(gaps) / len(gaps),
        "gap_gt3_ratio": None if not gaps else gap_gt3 / len(gaps),
        "token_correct_ratio": None if correctness_rows == 0 else correctness_hits / correctness_rows,
    }
    with (args.output_dir / "finalization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
