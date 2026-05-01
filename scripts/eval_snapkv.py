"""
SnapKV 评估脚本：使用压缩 KV Cache 的推理（实验组）
在 Pythia-70M 上测量压缩后的 PPL 和内存占用，并与 baseline 对比
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# 禁用 transformers 的依赖版本检查
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import transformers.utils.logging as hf_logging
hf_logging.disable_progress_bar()

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataset
from src.snapkv import SnapKVCompressor, apply_snapkv
from src.evaluate import measure_generation_time, measure_kv_cache_memory


@torch.no_grad()
def evaluate_ppl_with_snapkv(model, data_chunks, compressor, device="cuda"):
    """
    使用 SnapKV 压缩后评估困惑度。

    与 baseline 的区别：
        Baseline 使用完整的 KV Cache，每个位置都保留。
        SnapKV 先压缩 KV Cache，只保留重要位置，然后用压缩后的 KV 继续推理。

    为什么要这样评估？
        这模拟了真实使用场景：
        1. 用户输入一个长 prompt（例如一篇文档）
        2. 模型压缩 KV Cache 以节省内存
        3. 使用压缩后的 KV 继续生成回复
        我们需要验证：压缩后的 PPL 是否接近 baseline？如果接近，说明压缩没有损失太多信息。

    评估流程：
        1. 将每个序列分成前半段（prompt）和后半段（评估）
        2. 对前半段做 prefill + SnapKV 压缩
        3. 用压缩后的 KV Cache 作为前缀，对后半段做完整前向传播计算损失

    为什么要分成两段？
        - 前半段：模拟用户输入的长 prompt，用于生成和压缩 KV Cache
        - 后半段：用压缩后的 KV 作为上下文，计算后半段的预测损失
        这样可以测试：压缩后的 KV 是否还能准确预测后续内容

    参数：
        model: HuggingFace 因果语言模型
        data_chunks: input_ids 张量列表，每个形状 [1, seq_len]
        compressor: SnapKVCompressor 实例
        device: "cuda" 或 "cpu"

    返回：
        float: 困惑度值（越低越好，接近 baseline 说明压缩效果好）
    """
    model.eval()
    total_loss = 0.0  # 累计总损失
    total_tokens = 0  # 累计总 token 数

    for chunk in tqdm(data_chunks, desc="SnapKV PPL"):
        input_ids = chunk.to(device)
        seq_len = input_ids.size(1)  # 例如 2048

        # ====================================================================
        # 第 1 步：将序列一分为二
        # ====================================================================
        # 前半段用于 prefill + 压缩，后半段用于评估
        # 例如：seq_len=2048, split_pos=1024
        split_pos = seq_len // 2
        prompt_ids = input_ids[:, :split_pos]   # 前半段：位置 0-1023，作为 prompt
        eval_ids = input_ids[:, split_pos:]     # 后半段：位置 1024-2047，用于评估

        # ====================================================================
        # 第 2 步：对前半段执行 prefill 并压缩 KV Cache
        # ====================================================================
        # apply_snapkv 会：
        # 1. 将 prompt_ids 送入模型，获取完整的 KV Cache 和注意力权重
        # 2. 根据注意力模式压缩 KV Cache
        # 3. 返回压缩后的 KV Cache
        compressed_kv, _ = apply_snapkv(model, prompt_ids, compressor, device=device)
        # compressed_kv 是压缩后的 KV Cache，长度从 1024 压缩到约 160
        # （max_capacity=128 + observation_window=32）

        # ====================================================================
        # 第 3 步：使用压缩后的 KV Cache 处理后半段序列
        # ====================================================================
        # 直接用压缩后的 KV 作为前缀，对后半段做完整的前向传播
        # 这样可以正确计算损失，同时测试压缩后的 KV 是否保留了足够的信息
        outputs = model(
            input_ids=eval_ids,
            past_key_values=compressed_kv,
            labels=eval_ids,  # 用自身作为标签计算损失
        )

        # 累计损失（后半段有 eval_ids.size(1) - 1 个有效预测）
        loss_val = outputs.loss.item()
        if not (np.isnan(loss_val) or np.isinf(loss_val)):
            total_loss += loss_val * (eval_ids.size(1) - 1)
            total_tokens += eval_ids.size(1) - 1

    # ====================================================================
    # 第 4 步：计算平均损失和困惑度
    # ====================================================================
    if total_tokens == 0:
        print("警告: 没有有效的 token 用于计算 PPL")
        return float('inf')

    avg_loss = total_loss / total_tokens
    print(f"总损失: {total_loss:.4f}, 总 tokens: {total_tokens}, 平均损失: {avg_loss:.4f}")

    # 防止溢出：如果 avg_loss 过大，直接返回 inf
    if avg_loss > 100:
        print(f"警告: 平均损失过大 ({avg_loss:.2f})，PPL 会溢出")
        return float('inf')

    ppl = np.exp(avg_loss)  # 困惑度 = e^(平均损失)
    return ppl


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="SnapKV evaluation")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m",
                        help="HuggingFace 模型名称")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "pg19"],
                        help="评估数据集")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    parser.add_argument("--gen_length", type=int, default=128,
                        help="速度测试时生成的 token 数量")
    parser.add_argument("--max_capacity", type=int, default=768,
                        help="每个 head 最多保留的 KV 位置数")
    parser.add_argument("--observation_window", type=int, default=64,
                        help="观察窗口大小")
    parser.add_argument("--kernel_size", type=int, default=5,
                        help="平滑池化核大小")
    parser.add_argument("--output", type=str, default="results/snapkv.json",
                        help="结果保存路径")
    args = parser.parse_args()

    print(f"设备: {args.device}")
    print(f"加载模型: {args.model}")

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # FP16 在长序列下数值不稳定，必须用 FP32
        attn_implementation="eager",  # 必须使用 eager 以支持 output_attentions
    ).to(args.device)

    print(f"模型已加载。参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 初始化 SnapKV 压缩器
    compressor = SnapKVCompressor(
        kernel_size=args.kernel_size,
        observation_window=args.observation_window,
        max_capacity=args.max_capacity,
    )
    print(f"SnapKV 配置: max_capacity={args.max_capacity}, "
          f"observation_window={args.observation_window}, kernel_size={args.kernel_size}")

    # 加载数据集
    print(f"\n加载数据集: {args.dataset}")
    data_chunks = get_dataset(args.dataset, tokenizer, max_length=args.max_length)

    # === 使用 SnapKV 评估 PPL ===
    print("\n--- PPL 评估 (SnapKV) ---")
    ppl = evaluate_ppl_with_snapkv(model, data_chunks, compressor, device=args.device)
    print(f"困惑度 (Perplexity): {ppl:.2f}")

    # === 测量 KV Cache 压缩率 ===
    # 使用前半段序列（1024 tokens）来测量压缩效果
    prompt = data_chunks[0][:, :args.max_length//2].to(args.device)

    # 压缩后的 KV Cache 大小
    compressed_kv, _ = apply_snapkv(model, prompt, compressor, device=args.device)
    kv_mem = measure_kv_cache_memory(compressed_kv)

    # 原始完整 KV Cache 大小（用于对比）
    with torch.no_grad():
        full_outputs = model(input_ids=prompt, use_cache=True)
        full_kv_mem = measure_kv_cache_memory(full_outputs.past_key_values)

    # 计算压缩率 = 压缩后大小 / 原始大小
    compression_ratio = kv_mem["memory_mb"] / full_kv_mem["memory_mb"] if full_kv_mem["memory_mb"] > 0 else 0
    print(f"\nKV Cache: {full_kv_mem['memory_mb']:.2f} MB -> {kv_mem['memory_mb']:.2f} MB "
          f"({compression_ratio:.1%})")

    # === 保存结果到 JSON ===
    results = {
        "method": "snapkv",
        "model": args.model,
        "dataset": args.dataset,
        "max_length": args.max_length,
        "max_capacity": args.max_capacity,
        "observation_window": args.observation_window,
        "kernel_size": args.kernel_size,
        "ppl": ppl,
        "kv_cache_memory_mb": kv_mem["memory_mb"],
        "full_kv_cache_memory_mb": full_kv_mem["memory_mb"],
        "compression_ratio": compression_ratio,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
