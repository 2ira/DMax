import json
import os
from pathlib import Path
from typing import Any

from data_formats import load_jsonl


def _load_outputs(path: str | Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def _score_gsm8k(rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    from dInfer.evaluations.val_gsm8k import evaluate_example

    total = min(len(rows), len(outputs))
    correct = 0
    details = []
    for idx in range(total):
        result, gold_candidates, pred_candidates = evaluate_example(
            rows[idx],
            {"answer": outputs[idx].get("generated_text", "")},
        )
        correct += int(result.correct)
        details.append(
            {
                "sample_index": idx,
                "correct": bool(result.correct),
                "method": result.method,
                "gold_candidates": gold_candidates,
                "pred_candidates": pred_candidates,
            }
        )
    return {
        "score_name": "accuracy",
        "score": None if total == 0 else correct / total,
        "num_scored": total,
        "per_sample": details,
    }


def _score_math500(rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    from dInfer.evaluations.val_math import evaluate_example

    total = min(len(rows), len(outputs))
    correct = 0
    details = []
    for idx in range(total):
        result, gold_candidates, pred_candidates = evaluate_example(
            rows[idx],
            {"answer": outputs[idx].get("generated_text", "")},
        )
        correct += int(result.correct)
        details.append(
            {
                "sample_index": idx,
                "correct": bool(result.correct),
                "method": result.method,
                "gold_candidates": gold_candidates,
                "pred_candidates": pred_candidates,
            }
        )
    return {
        "score_name": "accuracy",
        "score": None if total == 0 else correct / total,
        "num_scored": total,
        "per_sample": details,
    }


def _score_humaneval(rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    from dInfer.evaluations.tasks.humaneval.utils import pass_at_1

    predictions = [row.get("generated_text", "") for row in outputs[: len(rows)]]
    references = [row["test"] for row in rows[: len(predictions)]]
    score = pass_at_1(references, predictions)
    return {
        "score_name": "pass@1",
        "score": score,
        "num_scored": len(predictions),
        "per_sample": None,
    }


def _score_mbpp(rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    from dInfer.evaluations.tasks.mbpp_sanitized.utils import pass_at_1

    predictions = [row.get("generated_text", "") for row in outputs[: len(rows)]]
    references = [row["test_list"] for row in rows[: len(predictions)]]
    score = pass_at_1(references, predictions)
    return {
        "score_name": "pass@1",
        "score": score,
        "num_scored": len(predictions),
        "per_sample": None,
    }


TASK_SCORERS = {
    "gsm8k": _score_gsm8k,
    "math500": _score_math500,
    "humaneval": _score_humaneval,
    "mbpp": _score_mbpp,
}


def score_outputs(task: str, subset_path: str | Path, outputs_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    rows = load_jsonl(subset_path)
    outputs = _load_outputs(outputs_path)
    if task not in TASK_SCORERS:
        raise ValueError(f"No scorer registered for task: {task}")
    result = TASK_SCORERS[task](rows, outputs)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "score_details.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return result
