import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import REPO_ROOT


@dataclass
class RuntimeControls:
    batch_size: int = 1
    block_length: int = 32
    max_gen_len: int = 2048
    cache: str = "prefix"
    gpus: str = "0"
    tp_size: int | None = None
    master_port: int = 24667
    use_compile: bool = False
    use_bd: bool = True
    trace_topk: int = 8
    enable_trace: bool = True
    record_router: bool = False
    backend_early_stop: bool = True
    maximum_unroll: int = 4
    expected_tpf: int = 4


@dataclass
class ModelSpec:
    name: str
    path: str
    tokenizer_path: str | None = None


@dataclass
class DatasetSpec:
    name: str
    task: str
    subset_path: str
    max_samples: int | None = None


@dataclass
class MethodSpec:
    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSpec:
    name: str
    model: str
    dataset: str
    methods: list[str]
    runtime_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    output_root: str
    runtime: RuntimeControls
    models: dict[str, ModelSpec]
    datasets: dict[str, DatasetSpec]
    methods: dict[str, MethodSpec]
    experiments: list[ExperimentSpec]
    suite_path: Path


def _require_keys(payload: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {', '.join(missing)}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base: Path, raw: str) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    repo_resolved = (REPO_ROOT / path).resolve()
    if repo_resolved.exists() or raw.startswith("research/") or raw.startswith("dInfer/"):
        return str(repo_resolved)
    suite_resolved = (base / path).resolve()
    return str(suite_resolved)


def load_suite(path: str | Path) -> BenchmarkSuite:
    suite_path = Path(path).resolve()
    payload = _load_json(suite_path)
    base = suite_path.parent

    runtime_payload = payload.get("runtime", {})
    runtime = RuntimeControls(**runtime_payload)

    models: dict[str, ModelSpec] = {}
    for item in payload.get("models", []):
        _require_keys(item, ["name", "path"], "models[]")
        models[item["name"]] = ModelSpec(
            name=item["name"],
            path=_resolve_path(base, item["path"]),
            tokenizer_path=_resolve_path(base, item["tokenizer_path"]) if item.get("tokenizer_path") else None,
        )

    datasets: dict[str, DatasetSpec] = {}
    for item in payload.get("datasets", []):
        _require_keys(item, ["name", "task", "subset_path"], "datasets[]")
        datasets[item["name"]] = DatasetSpec(
            name=item["name"],
            task=item["task"],
            subset_path=_resolve_path(base, item["subset_path"]),
            max_samples=item.get("max_samples"),
        )

    methods: dict[str, MethodSpec] = {}
    for item in payload.get("methods", []):
        _require_keys(item, ["name", "family"], "methods[]")
        methods[item["name"]] = MethodSpec(
            name=item["name"],
            family=item["family"],
            params=dict(item.get("params", {})),
        )

    experiments: list[ExperimentSpec] = []
    for item in payload.get("experiments", []):
        _require_keys(item, ["name", "model", "dataset", "methods"], "experiments[]")
        experiments.append(
            ExperimentSpec(
                name=item["name"],
                model=item["model"],
                dataset=item["dataset"],
                methods=list(item["methods"]),
                runtime_overrides=dict(item.get("runtime_overrides", {})),
            )
        )

    output_root = _resolve_path(base, payload.get("output_root", "research/phase1/benchmarks"))
    return BenchmarkSuite(
        output_root=output_root,
        runtime=runtime,
        models=models,
        datasets=datasets,
        methods=methods,
        experiments=experiments,
        suite_path=suite_path,
    )
