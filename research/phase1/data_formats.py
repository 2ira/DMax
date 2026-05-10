import json
from pathlib import Path
from typing import Any


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


TASK_FORMATTERS = {
    "gsm8k": _gsm8k_prompt,
    "math500": _math500_prompt,
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
