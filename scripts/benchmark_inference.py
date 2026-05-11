"""
Benchmark dense inference against SnapKV on TTFT, TPOT, throughput, KV memory,
and estimated FLOPs.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataset
from src.evaluate import (
    estimate_generation_flops,
    get_cache_sequence_length,
    measure_generation_time,
    measure_kv_cache_memory,
    measure_snapkv_generation_time,
)
from src.snapkv import SnapKVCompressor, apply_snapkv


def _round_metrics(metrics):
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def main():
    parser = argparse.ArgumentParser(description="Dense vs SnapKV inference benchmark")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "pg19"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--prompt_length", type=int, default=1024)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--max_capacity", type=int, default=768)
    parser.add_argument("--observation_window", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results/inference_metrics.json")
    args = parser.parse_args()

    if args.prompt_length > args.max_length:
        raise ValueError("--prompt_length must be <= --max_length")

    print(f"Device: {args.device}")
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    print(f"Loading dataset: {args.dataset}")
    data_chunks = get_dataset(args.dataset, tokenizer, max_length=args.max_length)
    prompt = data_chunks[0][:, : args.prompt_length].to(args.device)

    compressor = SnapKVCompressor(
        kernel_size=args.kernel_size,
        observation_window=args.observation_window,
        max_capacity=args.max_capacity,
    )

    if args.warmup_runs > 0:
        warmup_gen_length = min(args.gen_length, 8)
        print(f"\n--- Warmup ({args.warmup_runs} run) ---")
        measure_generation_time(
            model,
            prompt,
            gen_length=warmup_gen_length,
            device=args.device,
            num_runs=args.warmup_runs,
        )
        measure_snapkv_generation_time(
            model,
            prompt,
            compressor,
            gen_length=warmup_gen_length,
            device=args.device,
            num_runs=args.warmup_runs,
        )

    print("\n--- Dense speed ---")
    dense_speed = measure_generation_time(
        model,
        prompt,
        gen_length=args.gen_length,
        device=args.device,
        num_runs=args.num_runs,
    )

    print("--- SnapKV speed ---")
    snapkv_speed = measure_snapkv_generation_time(
        model,
        prompt,
        compressor,
        gen_length=args.gen_length,
        device=args.device,
        num_runs=args.num_runs,
    )

    with torch.no_grad():
        dense_outputs = model(input_ids=prompt, use_cache=True)
        dense_kv = dense_outputs.past_key_values
        dense_kv_mem = measure_kv_cache_memory(dense_kv)

        compressed_kv, _ = apply_snapkv(model, prompt, compressor, device=args.device)
        snapkv_kv_mem = measure_kv_cache_memory(compressed_kv)

    dense_cache_len = get_cache_sequence_length(dense_kv)
    snapkv_cache_len = get_cache_sequence_length(compressed_kv)
    dense_flops = estimate_generation_flops(
        model.config,
        prompt_length=args.prompt_length,
        gen_length=args.gen_length,
        decode_cache_length=dense_cache_len,
    )
    snapkv_flops = estimate_generation_flops(
        model.config,
        prompt_length=args.prompt_length,
        gen_length=args.gen_length,
        decode_cache_length=snapkv_cache_len,
    )

    comparisons = {
        "ttft_ratio_dense_over_snapkv": dense_speed["ttft_ms"] / snapkv_speed["ttft_ms"],
        "tpot_speedup_dense_over_snapkv": dense_speed["tpot_ms"] / snapkv_speed["tpot_ms"],
        "throughput_speedup_snapkv_over_dense": (
            snapkv_speed["throughput_tok_per_sec"] / dense_speed["throughput_tok_per_sec"]
        ),
        "kv_memory_ratio": snapkv_kv_mem["memory_mb"] / dense_kv_mem["memory_mb"],
        "kv_memory_saving": 1.0 - snapkv_kv_mem["memory_mb"] / dense_kv_mem["memory_mb"],
        "decode_flops_ratio": snapkv_flops["decode_flops"] / dense_flops["decode_flops"],
        "decode_flops_saving": 1.0 - snapkv_flops["decode_flops"] / dense_flops["decode_flops"],
        "total_flops_ratio": snapkv_flops["total_flops"] / dense_flops["total_flops"],
    }

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and str(args.device).startswith("cuda") else None,
        "max_length": args.max_length,
        "prompt_length": args.prompt_length,
        "gen_length": args.gen_length,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "snapkv_config": {
            "max_capacity": args.max_capacity,
            "observation_window": args.observation_window,
            "kernel_size": args.kernel_size,
        },
        "dense": {
            "speed": _round_metrics(dense_speed),
            "kv_cache_memory_mb": round(dense_kv_mem["memory_mb"], 6),
            "cache_length": dense_cache_len,
            "flops": _round_metrics(dense_flops),
        },
        "snapkv": {
            "speed": _round_metrics(snapkv_speed),
            "kv_cache_memory_mb": round(snapkv_kv_mem["memory_mb"], 6),
            "cache_length": snapkv_cache_len,
            "flops": _round_metrics(snapkv_flops),
        },
        "comparison": _round_metrics(comparisons),
        "flops_note": (
            "Estimated model matmul FLOPs; excludes sampling, top-k selection, cache movement, "
            "and SnapKV compression bookkeeping."
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nMetric                  Dense        SnapKV")
    print(f"TTFT (ms)          {dense_speed['ttft_ms']:10.2f}  {snapkv_speed['ttft_ms']:10.2f}")
    print(f"TPOT (ms)          {dense_speed['tpot_ms']:10.2f}  {snapkv_speed['tpot_ms']:10.2f}")
    print(f"Throughput tok/s   {dense_speed['throughput_tok_per_sec']:10.2f}  {snapkv_speed['throughput_tok_per_sec']:10.2f}")
    print(f"KV cache MB        {dense_kv_mem['memory_mb']:10.2f}  {snapkv_kv_mem['memory_mb']:10.2f}")
    print(f"Total GFLOPs       {dense_flops['total_gflops']:10.2f}  {snapkv_flops['total_gflops']:10.2f}")
    print(f"Avg GFLOPs/token   {dense_flops['avg_gflops_per_output_token']:10.2f}  {snapkv_flops['avg_gflops_per_output_token']:10.2f}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
