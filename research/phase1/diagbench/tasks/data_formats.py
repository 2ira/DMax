import json
from pathlib import Path
from typing import Any


def _first_present(row: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


def _normalize_choice_letter(text: str) -> str:
    cleaned = str(text or "").strip().upper()
    return cleaned[0] if cleaned[:1] in {"A", "B", "C", "D"} else ""


def _build_gpqa_choices(row: dict[str, Any]) -> tuple[list[tuple[str, str]], str]:
    if isinstance(row.get("choices"), list) and len(row["choices"]) >= 4:
        raw_choices = [str(item) for item in row["choices"][:4]]
        gold = _first_present(row, ["expected_answer", "answer", "correct_answer"])
        letter = _normalize_choice_letter(gold)
        if letter:
            return list(zip(["A", "B", "C", "D"], raw_choices)), letter
        if gold:
            for idx, choice in enumerate(raw_choices):
                if choice.strip() == gold.strip():
                    return list(zip(["A", "B", "C", "D"], raw_choices)), ["A", "B", "C", "D"][idx]
        return list(zip(["A", "B", "C", "D"], raw_choices)), ""

    if isinstance(row.get("options"), list) and len(row["options"]) >= 4:
        raw_choices = [str(item) for item in row["options"][:4]]
        gold = _first_present(row, ["expected_answer", "answer", "correct_answer"])
        letter = _normalize_choice_letter(gold)
        if letter:
            return list(zip(["A", "B", "C", "D"], raw_choices)), letter
        if gold:
            for idx, choice in enumerate(raw_choices):
                if choice.strip() == gold.strip():
                    return list(zip(["A", "B", "C", "D"], raw_choices)), ["A", "B", "C", "D"][idx]
        return list(zip(["A", "B", "C", "D"], raw_choices)), ""

    question_idx = int(row.get("_subset_index", 0)) % 4
    correct = _first_present(
        row,
        [
            "Correct Answer",
            "Extra Revised Correct Answer",
            "correct_answer",
            "answer",
            "expected_answer",
        ],
    )
    incorrects = [
        _first_present(row, ["Incorrect Answer 1", "Extra Revised Incorrect Answer 1", "incorrect_answer_1"]),
        _first_present(row, ["Incorrect Answer 2", "Extra Revised Incorrect Answer 2", "incorrect_answer_2"]),
        _first_present(row, ["Incorrect Answer 3", "Extra Revised Incorrect Answer 3", "incorrect_answer_3"]),
    ]
    rotation = [question_idx % 4, (question_idx + 1) % 4, (question_idx + 2) % 4, (question_idx + 3) % 4]
    slots = ["", "", "", ""]
    slots[rotation[0]] = correct
    for source_idx, target_idx in enumerate(rotation[1:]):
        slots[target_idx] = incorrects[source_idx]
    choices = list(zip(["A", "B", "C", "D"], slots))
    gold_letter = ["A", "B", "C", "D"][rotation[0]] if correct else ""
    return choices, gold_letter


def format_aime24_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    problem = _first_present(row, ["problem", "Problem", "question", "Question"])
    target = _first_present(row, ["answer", "Answer", "final_answer"])
    prompt = f"{problem}\nLet's think step by step, and put your final answer within \\boxed{{}}."
    meta = {}
    for src_key, dst_key in [
        ("solution", "solution"),
        ("Solution", "solution"),
        ("id", "id"),
        ("ID", "id"),
        ("year", "year"),
        ("url", "url"),
    ]:
        if src_key in row and dst_key not in meta:
            meta[dst_key] = row[src_key]
    return prompt, target, meta


def format_gpqa_cot_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if row.get("input_query"):
        prompt = str(row["input_query"])
        target = _first_present(
            row,
            [
                "expected_answer",
                "Correct Answer",
                "Extra Revised Correct Answer",
                "correct_answer",
                "answer",
            ],
        )
        meta = {
            "expected_answer": target,
            "record_id": row.get("Record ID", row.get("record_id")),
            "subdomain": row.get("Subdomain", row.get("subdomain")),
            "high_level_domain": row.get("High-level domain", row.get("high_level_domain")),
        }
        return prompt, target, meta

    question = _first_present(
        row,
        [
            "Question",
            "question",
            "Extra Revised Question",
            "Pre-Revision Question",
        ],
    )
    choices, gold_letter = _build_gpqa_choices(row)
    choices_text = "\n".join(f"{label}. {text}" for label, text in choices if str(text).strip())
    prompt = (
        "Answer the following multiple-choice question. Think step by step, then give your final answer as a single letter.\n\n"
        f"Question: {question}\n\n"
        f"{choices_text}\n\n"
        "Final answer:"
    )
    meta = {
        "correct_letter": gold_letter,
        "record_id": row.get("Record ID", row.get("record_id")),
        "subdomain": row.get("Subdomain", row.get("subdomain")),
        "high_level_domain": row.get("High-level domain", row.get("high_level_domain")),
        "choices": choices,
    }
    return prompt, gold_letter, meta


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    whole_enc = tokenizer(context + continuation)["input_ids"]
    context_enc = tokenizer(context)["input_ids"]
    continuation_enc = whole_enc[len(context_enc):]
    return context_enc, continuation_enc


def _gsm8k_prompt(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    prompt = f"Question: {row['question']}\nLet's think step by step\nAnswer:"
    target = row["answer"]
    return prompt, target, {}


def _math500_prompt(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    prompt = f"{row['problem']}\nLet's think step by step, and put your final answer within \\boxed{{}}."
    target = row.get("answer", "")
    meta = {}
    if "solution" in row:
        meta["solution"] = row["solution"]
    return prompt, target, meta


def _humaneval_prompt(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    prompt = (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        "```python\n"
        f"{row['prompt']}\n"
        "```\n\n"
        "Please enclose your code within delimiters as follows:\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
    )
    target = row.get("canonical_solution", "")
    meta = {
        "entry_point": row.get("entry_point"),
        "test": row.get("test"),
    }
    return prompt, target, meta


def _mbpp_prompt(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    tests = row.get("test_list", [])
    test_list_str = "\n".join(tests)
    prompt = (
        "You are an expert Python programmer, and here is your task: "
        f"{row['text']} Your code should pass these tests:\n\n"
        f"{test_list_str}\n\n"
        "Please enclose your code within delimiters as follows:\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
    )
    target = row.get("code", "")
    meta = {
        "test_list": tests,
        "test_list_str": test_list_str,
    }
    return prompt, target, meta


def _aime24_prompt(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return format_aime24_row(row)


def _gpqa_cot_prompt(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return format_gpqa_cot_row(row)


TASK_FORMATTERS = {
    "gsm8k": _gsm8k_prompt,
    "math500": _math500_prompt,
    "aime24": _aime24_prompt,
    "gpqa_cot": _gpqa_cot_prompt,
    "humaneval": _humaneval_prompt,
    "mbpp": _mbpp_prompt,
}


def prepare_samples(task: str, rows: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    formatter = TASK_FORMATTERS[task]
    prepared = []
    for idx, row in enumerate(rows):
        prompt_text, target_text, meta = formatter(row)
        prompt_ids, target_ids = encode_pair(tokenizer, prompt_text, target_text)
        prepared.append(
            {
                "sample_id": f"{task}-{idx}",
                "subset_index": row.get("_subset_index", idx),
                "task": task,
                "prompt_text": prompt_text,
                "target_text": target_text,
                "prompt_ids": prompt_ids,
                "target_ids": target_ids,
                "metadata": meta,
            }
        )
    return prepared
