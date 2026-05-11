import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

from ..core.benchmark_core import MethodRunConfig, launch_method_run
from ..core.benchmark_spec import RuntimeControls, load_suite
from ..tasks.data_formats import load_jsonl, prepare_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Run a controlled dLLM benchmark suite.")
    parser.add_argument("--suite", type=Path, required=True, help="Path to a benchmark suite JSON.")
    parser.add_argument("--experiments", nargs="*", default=None, help="Optional subset of experiment names.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset of method names.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _merge_runtime(base: RuntimeControls, overrides: dict) -> RuntimeControls:
    payload = dict(base.__dict__)
    payload.update(overrides)
    return RuntimeControls(**payload)


def _write_suite_manifest(output_root: Path, suite_path: Path, payload: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "suite_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _append_suite_row(output_root: Path, row: dict) -> None:
    csv_path = output_root / "suite_results.csv"
    fieldnames = [
        "experiment",
        "model",
        "dataset",
        "method",
        "score_name",
        "Score",
        "score",
        "TPF",
        "tpf",
        "TPS",
        "tps",
        "NFE",
        "total_nfe",
        "premature_finalization_rate",
        "mean_self_finalization_gap",
        "output_dir",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = parse_args()
    suite = load_suite(args.suite)
    selected_experiments = set(args.experiments or [])
    selected_methods = set(args.methods or [])
    output_root = Path(suite.output_root)

    manifest = {
        "suite_path": str(suite.suite_path),
        "output_root": str(output_root),
        "shared_runtime": suite.runtime.__dict__,
        "models": {name: spec.__dict__ for name, spec in suite.models.items()},
        "datasets": {name: spec.__dict__ for name, spec in suite.datasets.items()},
        "methods": {name: {"family": spec.family, "params": spec.params} for name, spec in suite.methods.items()},
        "experiments": [exp.__dict__ for exp in suite.experiments],
    }
    _write_suite_manifest(output_root, suite.suite_path, manifest)

    for experiment in suite.experiments:
        if selected_experiments and experiment.name not in selected_experiments:
            continue
        model_spec = suite.models[experiment.model]
        dataset_spec = suite.datasets[experiment.dataset]
        runtime = _merge_runtime(suite.runtime, experiment.runtime_overrides)
        prepared_samples = None

        model_path = Path(model_spec.path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path}\n"
                "Expected a local Hugging Face checkpoint directory with config/tokenizer/model files."
            )
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(
                f"Model directory is missing config.json: {model_path}\n"
                "This usually means the model download is incomplete or the suite path points to the wrong directory."
            )

        for method_name in experiment.methods:
            if selected_methods and method_name not in selected_methods:
                continue
            method_spec = suite.methods[method_name]
            run_output_dir = output_root / experiment.name / method_name
            if args.dry_run:
                print(json.dumps(
                    {
                        "experiment": experiment.name,
                        "method": method_name,
                        "model": model_spec.path,
                        "dataset": dataset_spec.subset_path,
                        "output_dir": str(run_output_dir),
                        "runtime": runtime.__dict__,
                        "method_family": method_spec.family,
                        "method_params": method_spec.params,
                    },
                    ensure_ascii=False,
                    indent=2,
                ))
                continue
            if prepared_samples is None:
                tokenizer = AutoTokenizer.from_pretrained(model_spec.tokenizer_path or model_spec.path, trust_remote_code=True)
                rows = load_jsonl(dataset_spec.subset_path)
                if dataset_spec.max_samples is not None:
                    rows = rows[: dataset_spec.max_samples]
                prepared_samples = prepare_samples(dataset_spec.task, rows, tokenizer)
            run_config = MethodRunConfig(
                experiment_name=experiment.name,
                model_name=model_spec.name,
                model_path=model_spec.path,
                tokenizer_path=model_spec.tokenizer_path or model_spec.path,
                task=dataset_spec.task,
                subset_path=dataset_spec.subset_path,
                output_dir=str(run_output_dir),
                method_name=method_name,
                method_family=method_spec.family,
                method_params=method_spec.params,
                runtime=runtime,
                prepared_samples=prepared_samples,
            )
            launch_method_run(run_config)
            summary_path = run_output_dir / "summary.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
                _append_suite_row(
                    output_root,
                    {
                        "experiment": experiment.name,
                        "model": model_spec.name,
                        "dataset": dataset_spec.name,
                        "method": method_name,
                        "score_name": summary.get("score_name"),
                        "Score": summary.get("Score", summary.get("score")),
                        "score": summary.get("score"),
                        "TPF": summary.get("TPF", summary.get("tpf")),
                        "tpf": summary.get("tpf"),
                        "TPS": summary.get("TPS", summary.get("tps")),
                        "tps": summary.get("tps"),
                        "NFE": summary.get("NFE", summary.get("total_nfe")),
                        "total_nfe": summary.get("total_nfe"),
                        "premature_finalization_rate": summary.get("premature_finalization_rate"),
                        "mean_self_finalization_gap": summary.get("mean_self_finalization_gap"),
                        "output_dir": str(run_output_dir),
                    },
                )


if __name__ == "__main__":
    main()
