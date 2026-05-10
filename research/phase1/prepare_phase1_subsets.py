import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_SPECS = {
    "gsm8k_200": {
        "loader": "hf",
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "limit": 200,
    },
    "math500_200": {
        "loader": "hf",
        "path": "HuggingFaceH4/MATH-500",
        "name": None,
        "split": "test",
        "limit": 200,
    },
    "humaneval_164": {
        "loader": "hf",
        "path": "openai/openai_humaneval",
        "name": None,
        "split": "test",
        "limit": 164,
    },
    "mbpp_sanitized_200": {
        "loader": "json_url",
        "path": "json",
        "name": None,
        "split": "train",
        "limit": 200,
        "data_files": "https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/data/sanitized-mbpp.json",
    },
    "aime24_30": {
        "loader": "hf",
        "path": "HuggingFaceH4/aime_2024",
        "name": "default",
        "split": "train",
        "limit": 30,
    },
    "gpqa_cot_100": {
        "loader": "hf",
        "path": "llamastack/gpqa_0shot_cot",
        "name": "gpqa_diamond",
        "split": "train",
        "limit": 100,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic phase-1 subsets for dLLM diagnostics.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where subset jsonl files and the manifest will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used before taking the first N examples.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable deterministic shuffling before subsetting.",
    )
    return parser.parse_args()


def load_spec_dataset(spec: dict[str, Any]):
    if spec["loader"] == "hf":
        return load_dataset(spec["path"], spec["name"], split=spec["split"])
    if spec["loader"] == "json_url":
        return load_dataset(spec["path"], data_files=spec["data_files"], split=spec["split"])
    raise ValueError(f"Unsupported loader: {spec['loader']}")


def subset_records(dataset, limit: int, seed: int, no_shuffle: bool):
    ds = dataset if no_shuffle else dataset.shuffle(seed=seed)
    limit = min(limit, len(ds))
    return ds.select(range(limit))


def write_jsonl(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(records):
            payload = dict(row)
            payload["_subset_index"] = idx
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    subset_dir = args.output_root / "subsets"
    manifest_dir = args.output_root / "manifests"
    subset_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "seed": args.seed,
        "shuffle_enabled": not args.no_shuffle,
        "datasets": {},
    }

    for subset_name, spec in DATASET_SPECS.items():
        dataset = load_spec_dataset(spec)
        subset = subset_records(dataset, spec["limit"], args.seed, args.no_shuffle)
        output_path = subset_dir / f"{subset_name}.jsonl"
        write_jsonl(subset, output_path)

        manifest["datasets"][subset_name] = {
            "source_path": spec["path"],
            "source_name": spec["name"],
            "source_split": spec["split"],
            "limit": spec["limit"],
            "num_written": len(subset),
            "output_path": str(output_path),
        }

        print(f"[phase1] wrote {len(subset)} rows -> {output_path}")

    manifest_path = manifest_dir / "phase1_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[phase1] wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
