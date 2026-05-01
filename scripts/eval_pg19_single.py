"""
PG-19 单样本评估：测试超长文本场景
"""

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_pg19
from src.snapkv import SnapKVCompressor, apply_snapkv
from src.evaluate import measure_kv_cache_memory


@torch.no_grad()
def evaluate_ppl_fair(model, data_chunks, device="cuda", desc="PPL"):
    """公平对比：前半段作为 prompt，只评估后半段的 PPL"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for chunk in tqdm(data_chunks, desc=desc):
        input_ids = chunk.to(device)
        seq_len = input_ids.size(1)

        split_pos = seq_len // 2
        prompt_ids = input_ids[:, :split_pos]
        eval_ids = input_ids[:, split_pos:]

        prompt_outputs = model(input_ids=prompt_ids, use_cache=True)
        past_kv = prompt_outputs.past_key_values

        outputs = model(
            input_ids=eval_ids,
            past_key_values=past_kv,
            labels=eval_ids,
        )

        loss_val = outputs.loss.item()
        if not (np.isnan(loss_val) or np.isinf(loss_val)):
            total_loss += loss_val * (eval_ids.size(1) - 1)
            total_tokens += eval_ids.size(1) - 1

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    return ppl


@torch.no_grad()
def evaluate_ppl_with_snapkv(model, data_chunks, compressor, device="cuda", desc="SnapKV PPL"):
    """SnapKV 评估"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for chunk in tqdm(data_chunks, desc=desc):
        input_ids = chunk.to(device)
        seq_len = input_ids.size(1)

        split_pos = seq_len // 2
        prompt_ids = input_ids[:, :split_pos]
        eval_ids = input_ids[:, split_pos:]

        compressed_kv, _ = apply_snapkv(model, prompt_ids, compressor, device=device)

        outputs = model(
            input_ids=eval_ids,
            past_key_values=compressed_kv,
            labels=eval_ids,
        )

        loss_val = outputs.loss.item()
        if not (np.isnan(loss_val) or np.isinf(loss_val)):
            total_loss += loss_val * (eval_ids.size(1) - 1)
            total_tokens += eval_ids.size(1) - 1

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    return ppl


def main():
    parser = argparse.ArgumentParser(description="PG-19 single sample evaluation")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_capacity", type=int, default=768)
    parser.add_argument("--observation_window", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results/pg19_single.json")
    args = parser.parse_args()

    print(f"设备: {args.device}")
    print(f"加载模型: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device)

    print(f"模型已加载。参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    print(f"\n加载 PG-19 数据集（单样本）")
    data_chunks = load_pg19(tokenizer, max_length=args.max_length, num_samples=1)
    print(f"加载了 {len(data_chunks)} 个数据块")

    # Baseline
    print("\n--- Baseline PPL ---")
    baseline_ppl = evaluate_ppl_fair(model, data_chunks, device=args.device, desc="Baseline")
    print(f"Baseline PPL: {baseline_ppl:.2f}")

    # SnapKV
    compressor = SnapKVCompressor(
        kernel_size=5,
        observation_window=args.observation_window,
        max_capacity=args.max_capacity,
    )
    print(f"\n--- SnapKV PPL (k={args.max_capacity}) ---")
    snapkv_ppl = evaluate_ppl_with_snapkv(model, data_chunks, compressor, device=args.device)
    print(f"SnapKV PPL: {snapkv_ppl:.2f}")

    # 测量 KV Cache
    prompt = data_chunks[0][:, :args.max_length//2].to(args.device)
    with torch.no_grad():
        full_outputs = model(input_ids=prompt, use_cache=True)
        full_kv_mem = measure_kv_cache_memory(full_outputs.past_key_values)

        compressed_kv, _ = apply_snapkv(model, prompt, compressor, device=args.device)
        comp_kv_mem = measure_kv_cache_memory(compressed_kv)

    compression_ratio = comp_kv_mem["memory_mb"] / full_kv_mem["memory_mb"]
    print(f"\nKV Cache: {full_kv_mem['memory_mb']:.2f} MB -> {comp_kv_mem['memory_mb']:.2f} MB ({compression_ratio:.1%})")

    results = {
        "dataset": "pg19_single_sample",
        "model": args.model,
        "max_length": args.max_length,
        "num_chunks": len(data_chunks),
        "baseline_ppl": baseline_ppl,
        "snapkv_ppl": snapkv_ppl,
        "max_capacity": args.max_capacity,
        "observation_window": args.observation_window,
        "full_kv_cache_mb": full_kv_mem["memory_mb"],
        "compressed_kv_cache_mb": comp_kv_mem["memory_mb"],
        "compression_ratio": compression_ratio,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
