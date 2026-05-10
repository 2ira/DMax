import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from .. import REPO_ROOT
from .benchmark_spec import RuntimeControls
from ..methods.minimal_decoders import build_decoder_from_method
from ..tasks.scorers import score_outputs
from .trace_metrics import write_trace_metrics


@dataclass
class MethodRunConfig:
    experiment_name: str
    model_name: str
    model_path: str
    tokenizer_path: str
    task: str
    subset_path: str
    output_dir: str
    method_name: str
    method_family: str
    method_params: dict
    runtime: RuntimeControls
    prepared_samples: list[dict]


def get_bucket_length(total_length: int, block_length: int) -> int:
    return block_length * math.ceil(total_length / block_length)


def build_generation_length(batch_samples: list[dict], block_length: int, max_gen_len: int) -> int:
    max_prompt = max(len(sample["prompt_ids"]) for sample in batch_samples)
    max_target = max(max(len(sample["target_ids"]), 1) for sample in batch_samples)
    total_length = get_bucket_length(max_prompt + max_target, block_length)
    return min(max_gen_len, max(total_length - max_prompt, block_length))


def build_batch_input(batch_samples: list[dict], mask_id: int, device: torch.device) -> torch.Tensor:
    max_prompt = max(len(sample["prompt_ids"]) for sample in batch_samples)
    batch = torch.full((len(batch_samples), max_prompt), mask_id, dtype=torch.long, device=device)
    for idx, sample in enumerate(batch_samples):
        prompt_ids = torch.tensor(sample["prompt_ids"], dtype=torch.long, device=device)
        batch[idx, : prompt_ids.shape[0]] = prompt_ids
    return batch


def build_sample_infos(batch_samples: list[dict], model_name: str, task: str, decoder_name: str) -> list[dict]:
    infos = []
    for sample in batch_samples:
        infos.append(
            {
                "sample_id": sample["sample_id"],
                "subset_index": sample["subset_index"],
                "task": task,
                "model_name": model_name,
                "prompt_text": sample["prompt_text"],
                "target_text": sample["target_text"],
                "prompt_ids": sample["prompt_ids"],
                "target_ids": sample["target_ids"],
                "prompt_length": len(sample["prompt_ids"]),
                "target_length": len(sample["target_ids"]),
                "metadata": sample.get("metadata", {}),
                "decoder": decoder_name,
            }
        )
    return infos


def _write_controls(path: Path, run_config: MethodRunConfig) -> None:
    controls = {
        "experiment_name": run_config.experiment_name,
        "model_name": run_config.model_name,
        "model_path": run_config.model_path,
        "task": run_config.task,
        "subset_path": run_config.subset_path,
        "method_name": run_config.method_name,
        "method_family": run_config.method_family,
        "method_params": run_config.method_params,
        "runtime": asdict(run_config.runtime),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(controls, f, ensure_ascii=False, indent=2)
        f.write("\n")


def worker(world_size: int, rank: int, gpu_id: int, config: MethodRunConfig):
    from dinfer import BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory
    from dinfer.decoding.diffusion_runner import ModelRunner
    from dinfer.decoding.trace_recorder import TraceRecorder
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from sglang.srt import distributed
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.server_args import ServerArgs

    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(config.runtime.master_port)
    distributed.init_distributed_environment(world_size, rank, "env://", rank, "nccl")
    distributed.initialize_model_parallel(config.runtime.tp_size, config.runtime.tp_size, 1, backend="nccl")

    model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
    server_args = ServerArgs(
        model_path=config.model_path,
        enable_dp_attention=True,
        trust_remote_code=True,
        tp_size=config.runtime.tp_size,
        dp_size=1,
        pp_size=1,
    )
    initialize_dp_attention(server_args=server_args, model_config=model_config)
    initialize_moe_config(server_args)
    model = LLaDA2SGLangLM(config=model_config, expert_map_path=".").eval()
    torch.set_default_dtype(torch.bfloat16)
    model.load_weights(config.model_path, device=device)
    initialize_moe_config(server_args)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
    max_prompt_len = max(len(sample["prompt_ids"]) for sample in config.prepared_samples)
    max_target_len = max(max(len(sample["target_ids"]), 1) for sample in config.prepared_samples)
    max_length = get_bucket_length(max_prompt_len + max_target_len, config.runtime.block_length)
    model = ModelRunner(
        model,
        device,
        server_args=server_args,
        max_length=max_length,
        block_length=config.runtime.block_length,
        enable_compile=config.runtime.use_compile,
    )

    mask_id = 156895
    eos_id = 156892
    decoder = build_decoder_from_method(
        config.method_name,
        type("MethodProxy", (), {"family": config.method_family, "params": config.method_params}),
        mask_id=mask_id,
        eos_id=eos_id,
    )
    decoder.trace_collect_router = config.runtime.record_router
    decoder.trace_topk = config.runtime.trace_topk

    cache_factory = KVCacheFactory(
        config.runtime.cache,
        is_bd_model=config.runtime.use_bd,
        backend="sglang",
        max_length=max_length,
    )
    dllm = BlockDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(start_block_align=True, use_block_diffusion=config.runtime.use_bd),
        cache_factory=cache_factory,
        early_stop=config.runtime.backend_early_stop,
        maximum_unroll=config.runtime.maximum_unroll,
        expected_tpf=config.runtime.expected_tpf,
        backend="sglang",
    )

    trace_recorder = None
    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_controls(output_dir / "controls.json", config)
        outputs_fh = (output_dir / "outputs.jsonl").open("w", encoding="utf-8")
        if config.runtime.enable_trace:
            trace_path = output_dir / "traces.jsonl"
            trace_recorder = TraceRecorder(
                trace_path,
                topk=config.runtime.trace_topk,
                record_router=config.runtime.record_router,
            )
            dllm.set_trace_recorder(trace_recorder)
    else:
        outputs_fh = None

    batch_latencies_ms = []
    batch_nfes = []
    batch_tpfs = []
    total_tokens = 0
    total_nfe = 0
    total_time_s = 0.0

    for batch_start in range(0, len(config.prepared_samples), config.runtime.batch_size):
        batch_samples = config.prepared_samples[batch_start : batch_start + config.runtime.batch_size]
        input_ids = build_batch_input(batch_samples, mask_id=mask_id, device=device)
        gen_length = build_generation_length(batch_samples, config.runtime.block_length, config.runtime.max_gen_len)

        if rank == 0 and trace_recorder is not None:
            trace_recorder.start_batch(build_sample_infos(batch_samples, config.model_name, config.task, config.method_name))

        start = time.perf_counter()
        prev_forwards = dllm.num_forwards
        out = dllm.generate(input_ids, gen_length=gen_length, block_length=config.runtime.block_length)
        nfe = dllm.num_forwards - prev_forwards
        elapsed_s = time.perf_counter() - start

        if rank == 0 and trace_recorder is not None:
            trace_recorder.finish_batch(out)

        batch_tokens = 0
        per_sample_records = []
        for idx, sample in enumerate(batch_samples):
            answer_ids = out[idx, len(sample["prompt_ids"]):].detach().cpu()
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
            gen_tokens = int(((out[idx] != eos_id) & (out[idx] != mask_id)).sum().item() - len(sample["prompt_ids"]))
            batch_tokens += max(gen_tokens, 0)
            per_sample_records.append(
                {
                    "sample_id": sample["sample_id"],
                    "subset_index": sample["subset_index"],
                    "task": config.task,
                    "model_name": config.model_name,
                    "decoder": config.method_name,
                    "generated_text": answer_text,
                    "generated_ids": answer_ids.tolist(),
                    "generated_length": max(gen_tokens, 0),
                    "planned_gen_length": gen_length,
                    "nfe": nfe,
                    "latency_ms": elapsed_s * 1000.0,
                }
            )

        total_tokens += batch_tokens
        total_nfe += nfe
        total_time_s += elapsed_s
        batch_nfes.append(nfe)
        batch_latencies_ms.append(elapsed_s * 1000.0 / max(len(batch_samples), 1))
        batch_tpfs.append(batch_tokens / max(nfe, 1) / max(len(batch_samples), 1))

        if rank == 0:
            for record in per_sample_records:
                outputs_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            outputs_fh.flush()

    if outputs_fh is not None:
        outputs_fh.close()
    if trace_recorder is not None:
        trace_recorder.close()

    if rank == 0:
        summary = {
            "decoder": config.method_name,
            "num_samples": len(config.prepared_samples),
            "batch_size": config.runtime.batch_size,
            "total_nfe": total_nfe,
            "mean_nfe_per_batch": None if not batch_nfes else sum(batch_nfes) / len(batch_nfes),
            "total_generated_tokens": total_tokens,
            "tpf": None if total_nfe == 0 else total_tokens / total_nfe,
            "mean_tpf": None if not batch_tpfs else sum(batch_tpfs) / len(batch_tpfs),
            "median_sample_latency_ms": None if not batch_latencies_ms else statistics.median(batch_latencies_ms),
            "p90_sample_latency_ms": None if len(batch_latencies_ms) < 2 else sorted(batch_latencies_ms)[max(0, math.ceil(0.9 * len(batch_latencies_ms)) - 1)],
            "tps": None if total_time_s <= 0 else total_tokens / total_time_s,
            "fps": None if total_time_s <= 0 else total_nfe / total_time_s,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")


def launch_method_run(config: MethodRunConfig) -> None:
    gpus = [int(gpu) for gpu in config.runtime.gpus.split(";") if gpu.strip()]
    if config.runtime.tp_size is None:
        config.runtime.tp_size = len(gpus)

    if len(gpus) == 1:
        worker(1, 0, gpus[0], config)
    else:
        procs = []
        ctx = mp.get_context("spawn")
        for rank, gpu in enumerate(gpus):
            proc = ctx.Process(target=worker, args=(len(gpus), rank, gpu, config))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"worker exited with code {proc.exitcode}")

    outputs_path = Path(config.output_dir) / "outputs.jsonl"
    summary_path = Path(config.output_dir) / "summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

    score_payload = score_outputs(
        task=config.task,
        subset_path=config.subset_path,
        outputs_path=outputs_path,
        output_dir=config.output_dir,
    )
    summary["score_name"] = score_payload["score_name"]
    summary["score"] = score_payload["score"]
    summary["Score"] = score_payload["score"]
    summary["num_scored"] = score_payload["num_scored"]

    trace_path = Path(config.output_dir) / "traces.jsonl"
    if config.runtime.enable_trace and trace_path.exists():
        trace_payload = write_trace_metrics(trace_path, config.output_dir)
        summary["premature_finalization_rate"] = trace_payload["premature"]["premature_finalization_rate"]
        summary["mean_self_finalization_gap"] = trace_payload["self_finalization"]["mean_self_finalization_gap"]
    else:
        summary["premature_finalization_rate"] = None
        summary["mean_self_finalization_gap"] = None

    summary["TPF"] = summary.get("tpf")
    summary["TPS"] = summary.get("tps")
    summary["NFE"] = summary.get("total_nfe")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
