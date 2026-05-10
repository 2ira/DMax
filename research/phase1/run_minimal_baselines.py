import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "dInfer" / "python"
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "research" / "phase1" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "research" / "phase1" / ".cache"))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT / "research" / "phase1") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "research" / "phase1"))

from data_formats import load_jsonl, prepare_samples  # noqa: E402
from minimal_decoders import MinimalJotDecoder, MinimalSTDecDecoder  # noqa: E402


@dataclass
class BaselineConfig:
    model_path: str
    task: str
    output_dir: str
    decoder_name: str
    batch_size: int
    block_length: int
    threshold: float
    cache: str
    tp_size: int
    gpus: list[int]
    master_port: int
    use_compile: bool
    use_bd: bool
    prepared_samples: list[dict]
    trace_topk: int
    record_router: bool
    enable_trace: bool
    max_gen_len: int


def parse_args():
    parser = argparse.ArgumentParser(description="Run minimal Phase-1 baselines on frozen subsets.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", choices=["gsm8k", "math500", "aime24", "gpqa_cot", "humaneval", "mbpp"], required=True)
    parser.add_argument("--subset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder", choices=["standard", "credit", "jot", "stdec"], default="standard")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cache", default="prefix")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=24667)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-gen-len", type=int, default=2048)
    parser.add_argument("--trace-topk", type=int, default=8)
    parser.add_argument("--record-router", action="store_true")
    parser.add_argument("--enable-trace", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-bd", action="store_true")
    return parser.parse_args()


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


def build_sample_infos(batch_samples: list[dict], model_name: str, task: str) -> list[dict]:
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
                "decoder": None,
            }
        )
    return infos


def worker(world_size: int, rank: int, gpu_id: int, config: BaselineConfig):
    from dinfer import BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory, ThresholdParallelDecoder, CreditThresholdParallelDecoder
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
    os.environ["MASTER_PORT"] = str(config.master_port)
    distributed.init_distributed_environment(world_size, rank, "env://", rank, "nccl")
    distributed.initialize_model_parallel(config.tp_size, config.tp_size, 1, backend="nccl")

    model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
    server_args = ServerArgs(
        model_path=config.model_path,
        enable_dp_attention=True,
        trust_remote_code=True,
        tp_size=config.tp_size,
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

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    max_prompt_len = max(len(sample["prompt_ids"]) for sample in config.prepared_samples)
    max_target_len = max(max(len(sample["target_ids"]), 1) for sample in config.prepared_samples)
    max_length = get_bucket_length(max_prompt_len + max_target_len, config.block_length)
    model = ModelRunner(
        model,
        device,
        server_args=server_args,
        max_length=max_length,
        block_length=config.block_length,
        enable_compile=config.use_compile,
    )

    mask_id = 156895
    eos_id = 156892
    if config.decoder_name == "standard":
        decoder = ThresholdParallelDecoder(temperature=0, threshold=config.threshold, mask_id=mask_id, eos_id=eos_id)
    elif config.decoder_name == "credit":
        decoder = CreditThresholdParallelDecoder(temperature=0, threshold=config.threshold, mask_id=mask_id, eos_id=eos_id)
    elif config.decoder_name == "jot":
        decoder = MinimalJotDecoder(temperature=0, threshold=config.threshold, mask_id=mask_id, eos_id=eos_id)
    elif config.decoder_name == "stdec":
        decoder = MinimalSTDecDecoder(temperature=0, threshold=config.threshold, mask_id=mask_id, eos_id=eos_id)
    else:
        raise ValueError(config.decoder_name)

    decoder.trace_collect_router = config.record_router
    cache_factory = KVCacheFactory(config.cache, is_bd_model=config.use_bd, backend="sglang", max_length=max_length)
    dllm = BlockDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(start_block_align=True, use_block_diffusion=config.use_bd),
        cache_factory=cache_factory,
        early_stop=True,
        maximum_unroll=4,
        expected_tpf=4,
        backend="sglang",
    )

    trace_recorder = None
    if rank == 0 and config.enable_trace:
        trace_path = Path(config.output_dir) / "traces.jsonl"
        trace_recorder = TraceRecorder(trace_path, topk=config.trace_topk, record_router=config.record_router)
        dllm.set_trace_recorder(trace_recorder)

    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs_fh = (output_dir / "outputs.jsonl").open("w", encoding="utf-8")
    else:
        outputs_fh = None

    batch_latencies_ms = []
    batch_nfes = []
    batch_tpfs = []
    total_tokens = 0
    total_nfe = 0
    total_time_s = 0.0

    batches = range(0, len(config.prepared_samples), config.batch_size)
    for batch_start in batches:
        batch_samples = config.prepared_samples[batch_start : batch_start + config.batch_size]
        input_ids = build_batch_input(batch_samples, mask_id=mask_id, device=device)
        gen_length = build_generation_length(batch_samples, config.block_length, config.max_gen_len)

        if rank == 0 and trace_recorder is not None:
            infos = build_sample_infos(batch_samples, config.model_path, config.task)
            for info in infos:
                info["decoder"] = config.decoder_name
            trace_recorder.start_batch(infos)

        start = time.perf_counter()
        prev_forwards = dllm.num_forwards
        out = dllm.generate(input_ids, gen_length=gen_length, block_length=config.block_length)
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
                    "decoder": config.decoder_name,
                    "generated_text": answer_text,
                    "generated_ids": answer_ids.tolist(),
                    "generated_length": max(gen_tokens, 0),
                    "batch_elapsed_ms": elapsed_s * 1000.0,
                    "batch_nfe": nfe,
                }
            )
        total_tokens += batch_tokens
        total_nfe += nfe
        total_time_s += elapsed_s

        batch_latencies_ms.append((elapsed_s * 1000.0) / max(len(batch_samples), 1))
        batch_nfes.append(nfe)
        batch_tpfs.append(batch_tokens / max(nfe * max(len(batch_samples), 1), 1))

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
            "decoder": config.decoder_name,
            "num_samples": len(config.prepared_samples),
            "batch_size": config.batch_size,
            "total_nfe": total_nfe,
            "total_generated_tokens": total_tokens,
            "mean_tpf": None if not batch_tpfs else sum(batch_tpfs) / len(batch_tpfs),
            "median_sample_latency_ms": None if not batch_latencies_ms else statistics.median(batch_latencies_ms),
            "p90_sample_latency_ms": None if len(batch_latencies_ms) < 2 else sorted(batch_latencies_ms)[max(0, math.ceil(0.9 * len(batch_latencies_ms)) - 1)],
            "mean_fps": None if total_time_s <= 0 else total_nfe / total_time_s,
            "mean_tps": None if total_time_s <= 0 else total_tokens / total_time_s,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")


def main():
    args = parse_args()
    gpus = [int(gpu) for gpu in args.gpus.split(";") if gpu.strip()]
    tp_size = args.tp_size or len(gpus)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rows = load_jsonl(args.subset_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    prepared_samples = prepare_samples(args.task, rows, tokenizer)

    config = BaselineConfig(
        model_path=args.model_path,
        task=args.task,
        output_dir=str(args.output_dir),
        decoder_name=args.decoder,
        batch_size=args.batch_size,
        block_length=args.block_length,
        threshold=args.threshold,
        cache=args.cache,
        tp_size=tp_size,
        gpus=gpus,
        master_port=args.master_port,
        use_compile=not args.disable_compile,
        use_bd=not args.disable_bd,
        prepared_samples=prepared_samples,
        trace_topk=args.trace_topk,
        record_router=args.record_router,
        enable_trace=args.enable_trace,
        max_gen_len=args.max_gen_len,
    )

    if len(gpus) == 1:
        worker(1, 0, gpus[0], config)
        return

    procs = []
    for rank, gpu in enumerate(gpus):
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=worker, args=(len(gpus), rank, gpu, config))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"baseline worker exited with code {proc.exitcode}")


if __name__ == "__main__":
    main()
