import json
import os
import re
from pathlib import Path
from typing import Any

from .data_formats import format_gpqa_cot_row, load_jsonl


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


def _score_aime24(rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
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


def _extract_choice_letter(text: str) -> str:
    raw = str(text or "")
    patterns = [
        r"(?i)final answer\s*[:：]?\s*([ABCD])\b",
        r"(?i)answer\s*[:：]?\s*([ABCD])\b",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw)
        if matches:
            return matches[-1].upper()
    return ""


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _extract_prompt_choices(prompt: str) -> dict[str, str]:
    choice_map: dict[str, str] = {}
    for line in str(prompt or "").splitlines():
        match = re.match(r"^\s*([ABCD])[\.\):]\s*(.+?)\s*$", line.strip(), flags=re.IGNORECASE)
        if match:
            choice_map[match.group(1).upper()] = _normalize_text(match.group(2))
    return choice_map


def _score_gpqa_cot(rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    total = min(len(rows), len(outputs))
    correct = 0
    details = []

    for idx in range(total):
        prompt_text, target_text, meta = format_gpqa_cot_row(rows[idx])
        prediction = outputs[idx].get("generated_text", "")
        pred_letter = _extract_choice_letter(prediction)
        gold_letter = _extract_choice_letter(target_text) or _extract_choice_letter(meta.get("correct_letter", ""))
        gold_text = _normalize_text(meta.get("expected_answer", target_text))

        if not gold_letter and gold_text:
            choice_map = _extract_prompt_choices(prompt_text)
            for candidate_letter, candidate_text in choice_map.items():
                if candidate_text == gold_text:
                    gold_letter = candidate_letter
                    break

        if gold_letter:
            is_correct = pred_letter == gold_letter
            method = "choice_letter"
            gold_candidate = gold_letter
            pred_candidate = pred_letter
        else:
            is_correct = _normalize_text(prediction) == gold_text
            method = "normalized_text"
            gold_candidate = gold_text
            pred_candidate = _normalize_text(prediction)

        correct += int(is_correct)
        details.append(
            {
                "sample_index": idx,
                "correct": bool(is_correct),
                "method": method,
                "gold_candidate": gold_candidate,
                "pred_candidate": pred_candidate,
            }
        )

    return {
        "score_name": "accuracy",
        "score": None if total == 0 else correct / total,
        "num_scored": total,
        "per_sample": details,
    }


TASK_SCORERS = {
    "gsm8k": _score_gsm8k,
    "math500": _score_math500,
    "aime24": _score_aime24,
    "gpqa_cot": _score_gpqa_cot,
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
