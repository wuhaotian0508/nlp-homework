"""
Baseline 公平对比评估：使用与 SnapKV 相同的评估方式
前半段作为 prompt，只评估后半段的 PPL
"""

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataset
from src.evaluate import measure_kv_cache_memory


@torch.no_grad()
def evaluate_ppl_fair(model, data_chunks, device="cuda"):
    """
    公平对比：与 SnapKV 使用相同的评估方式
    前半段作为 prompt，只评估后半段的 PPL
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for chunk in tqdm(data_chunks, desc="Baseline PPL (fair)"):
        input_ids = chunk.to(device)
        seq_len = input_ids.size(1)

        # 与 SnapKV 相同：分成两段
        split_pos = seq_len // 2
        prompt_ids = input_ids[:, :split_pos]   # 前半段
        eval_ids = input_ids[:, split_pos:]     # 后半段

        # 对前半段做 prefill，获取完整 KV Cache
        prompt_outputs = model(input_ids=prompt_ids, use_cache=True)
        past_kv = prompt_outputs.past_key_values

        # 用完整 KV Cache 评估后半段
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
    print(f"总损失: {total_loss:.4f}, 总 tokens: {total_tokens}, 平均损失: {avg_loss:.4f}")
    ppl = np.exp(avg_loss)
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Baseline fair comparison")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "pg19"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results/baseline_fair.json")
    args = parser.parse_args()

    print(f"设备: {args.device}")
    print(f"加载模型: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # 与 SnapKV 相同，用 FP32
        attn_implementation="eager",  # 与 SnapKV 相同
    ).to(args.device)

    print(f"模型已加载。参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    print(f"\n加载数据集: {args.dataset}")
    data_chunks = get_dataset(args.dataset, tokenizer, max_length=args.max_length)

    print("\n--- PPL 评估 (公平对比) ---")
    ppl = evaluate_ppl_fair(model, data_chunks, device=args.device)
    print(f"困惑度 (Perplexity): {ppl:.2f}")

    # 测量 KV Cache 大小
    prompt = data_chunks[0][:, :256].to(args.device)
    with torch.no_grad():
        outputs = model(input_ids=prompt, use_cache=True)
        kv_mem = measure_kv_cache_memory(outputs.past_key_values)
    print(f"KV Cache 内存: {kv_mem['memory_mb']:.2f} MB")

    results = {
        "method": "dense_baseline_fair",
        "model": args.model,
        "dataset": args.dataset,
        "max_length": args.max_length,
        "ppl": ppl,
        "kv_cache_memory_mb": kv_mem["memory_mb"],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
