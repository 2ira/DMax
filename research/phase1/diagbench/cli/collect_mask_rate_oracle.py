import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from ..tasks.data_formats import load_jsonl, prepare_samples


@dataclass
class OracleConfig:
    model_path: str
    task: str
    output_path: str
    block_length: int
    tp_size: int
    gpus: list[int]
    master_port: int
    use_compile: bool
    prepared_samples: list[dict]
    tokenizer_name: str
    mask_rates: list[float]
    seed: int


def parse_args():
    parser = argparse.ArgumentParser(description="Collect one-step masked-token oracle accuracy for phase-1.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", choices=["gsm8k", "math500", "aime24", "gpqa_cot", "humaneval", "mbpp"], required=True)
    parser.add_argument("--subset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--master-port", type=int, default=24577)
    parser.add_argument("--mask-rates", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def get_bucket_length(total_length: int, block_length: int) -> int:
    return block_length * math.ceil(total_length / block_length)


def build_bd_attn_mask(batch_size: int, total_length: int, block_length: int, device: torch.device) -> torch.Tensor:
    num_blocks = total_length // block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))
    return block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)


def worker(world_size: int, rank: int, gpu_id: int, config: OracleConfig):
    from dinfer.decoding.diffusion_runner import ModelRunner
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

    max_total_len = max(len(s["prompt_ids"]) + max(len(s["target_ids"]), 1) for s in config.prepared_samples)
    max_total_len = get_bucket_length(max_total_len, config.block_length)
    model = ModelRunner(
        model,
        device,
        server_args=server_args,
        max_length=max_total_len,
        block_length=config.block_length,
        enable_compile=config.use_compile,
    )

    mask_id = 156895
    rng = random.Random(config.seed)
    results = []

    for sample in config.prepared_samples:
        prompt_ids = sample["prompt_ids"]
        target_ids = sample["target_ids"]
        if not target_ids:
            continue
        full_ids = prompt_ids + target_ids
        seq_len = get_bucket_length(len(full_ids), config.block_length)
        input_ids = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
        input_ids[0, : len(full_ids)] = torch.tensor(full_ids, dtype=torch.long, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        attn_mask = build_bd_attn_mask(1, seq_len, config.block_length, device)
        target_offset = len(prompt_ids)

        for mask_rate in config.mask_rates:
            num_to_mask = max(1, int(round(len(target_ids) * mask_rate)))
            target_positions = list(range(len(target_ids)))
            rng.shuffle(target_positions)
            masked_local = sorted(target_positions[:num_to_mask])
            masked_global = [target_offset + pos for pos in masked_local]

            masked_input = input_ids.clone()
            masked_input[0, masked_global] = mask_id
            output = model(
                masked_input[:, :seq_len],
                attention_mask=attn_mask[:, :seq_len, :seq_len],
                position_ids=position_ids[:, :seq_len],
                use_cache=False,
            )
            logits = output.logits[0, masked_global]
            pred = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            gold = [target_ids[pos] for pos in masked_local]
            correct = sum(int(p == g) for p, g in zip(pred, gold))
            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "task": config.task,
                    "mask_rate": mask_rate,
                    "num_masked": len(masked_local),
                    "correct": correct,
                    "accuracy": correct / len(masked_local),
                }
            )

    if rank == 0:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
            if results:
                writer.writeheader()
                writer.writerows(results)
        summary = {}
        for rate in config.mask_rates:
            subset = [r for r in results if r["mask_rate"] == rate]
            summary[str(rate)] = None if not subset else sum(r["accuracy"] for r in subset) / len(subset)
        with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
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
    mask_rates = [float(item) for item in args.mask_rates.split(",") if item.strip()]

    config = OracleConfig(
        model_path=args.model_path,
        task=args.task,
        output_path=str(args.output_path),
        block_length=args.block_length,
        tp_size=tp_size,
        gpus=gpus,
        master_port=args.master_port,
        use_compile=not args.disable_compile,
        prepared_samples=prepared_samples,
        tokenizer_name=args.model_path,
        mask_rates=mask_rates,
        seed=args.seed,
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
            raise RuntimeError(f"mask-rate oracle worker exited with code {proc.exitcode}")


if __name__ == "__main__":
    main()
