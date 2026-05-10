import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from ..tasks.data_formats import load_jsonl, prepare_samples


@dataclass
class CollectConfig:
    model_path: str
    task: str
    output_dir: str
    batch_size: int
    block_length: int
    threshold: float
    cache: str
    tp_size: int
    gpus: list[int]
    master_port: int
    use_compile: bool
    use_bd: bool
    model_type: str
    trace_topk: int
    record_router: bool
    max_gen_len: int
    prepared_samples: list[dict]
    tokenizer_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase-1 dLLM traces from the DMax decoding path.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", choices=["gsm8k", "math500", "aime24", "gpqa_cot", "humaneval", "mbpp"], required=True)
    parser.add_argument("--subset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cache", default="prefix")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=24567)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-gen-len", type=int, default=2048)
    parser.add_argument("--model-type", default="llada2")
    parser.add_argument("--trace-topk", type=int, default=8)
    parser.add_argument("--record-router", action="store_true")
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
            }
        )
    return infos

def worker(world_size: int, rank: int, gpu_id: int, config: CollectConfig):
    from dinfer import BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory, ThresholdParallelDecoder
    from dinfer.decoding.diffusion_runner import ModelRunner
    from dinfer.decoding.trace_recorder import TraceRecorder
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.server_args import ServerArgs
    from sglang.srt import distributed

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

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
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
    decoder = ThresholdParallelDecoder(temperature=0, threshold=config.threshold, mask_id=mask_id, eos_id=eos_id)
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

    recorder = None
    if rank == 0:
        trace_path = Path(config.output_dir) / "traces.jsonl"
        recorder = TraceRecorder(trace_path, topk=config.trace_topk, record_router=config.record_router)
        dllm.set_trace_recorder(recorder)
        decoder.trace_topk = config.trace_topk

    batches = range(0, len(config.prepared_samples), config.batch_size)
    outputs_path = Path(config.output_dir) / "outputs.jsonl"
    if rank == 0:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        outputs_fh = outputs_path.open("w", encoding="utf-8")
    else:
        outputs_fh = None

    for batch_start in batches:
        batch_samples = config.prepared_samples[batch_start : batch_start + config.batch_size]
        input_ids = build_batch_input(batch_samples, mask_id=mask_id, device=device)
        gen_length = build_generation_length(batch_samples, config.block_length, config.max_gen_len)

        if rank == 0 and recorder is not None:
            recorder.start_batch(build_sample_infos(batch_samples, config.model_path, config.task))

        out = dllm.generate(input_ids, gen_length=gen_length, block_length=config.block_length)

        if rank == 0 and recorder is not None:
            recorder.finish_batch(out)
            for idx, sample in enumerate(batch_samples):
                answer_ids = out[idx, len(sample["prompt_ids"]):].detach().cpu()
                answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
                outputs_fh.write(
                    json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            "subset_index": sample["subset_index"],
                            "generated_text": answer_text,
                            "generated_ids": answer_ids.tolist(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            outputs_fh.flush()

    if outputs_fh is not None:
        outputs_fh.close()
    if recorder is not None:
        recorder.close()


def main():
    args = parse_args()
    gpus = [int(gpu) for gpu in args.gpus.split(";") if gpu.strip()]
    tp_size = args.tp_size or len(gpus)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rows = load_jsonl(args.subset_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    prepared_samples = prepare_samples(args.task, rows, tokenizer)

    config = CollectConfig(
        model_path=args.model_path,
        task=args.task,
        output_dir=str(args.output_dir),
        batch_size=args.batch_size,
        block_length=args.block_length,
        threshold=args.threshold,
        cache=args.cache,
        tp_size=tp_size,
        gpus=gpus,
        master_port=args.master_port,
        use_compile=not args.disable_compile,
        use_bd=not args.disable_bd,
        model_type=args.model_type,
        trace_topk=args.trace_topk,
        record_router=args.record_router,
        max_gen_len=args.max_gen_len,
        prepared_samples=prepared_samples,
        tokenizer_name=args.model_path,
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
            raise RuntimeError(f"trace worker exited with code {proc.exitcode}")


if __name__ == "__main__":
    main()
