"""
评估工具模块：困惑度（PPL）计算和生成速度测量
"""

import time
import torch
import numpy as np
from tqdm import tqdm  # 进度条库


def _is_cuda_device(device):
    return str(device).startswith("cuda") and torch.cuda.is_available()


def _iter_cache_layers(past_key_values):
    """Support both legacy tuple caches and transformers DynamicCache."""
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            yield layer.keys, layer.values
        return
    if hasattr(past_key_values, "to_legacy_cache"):
        for layer_kv in past_key_values.to_legacy_cache():
            yield layer_kv[0], layer_kv[1]
        return
    for layer_kv in past_key_values:
        yield layer_kv[0], layer_kv[1]


@torch.no_grad()  # 禁用梯度计算，节省内存和加速推理
def evaluate_ppl(model, data_chunks, device="cuda", desc="Evaluating PPL"):
    """
    在一系列分词后的数据块上计算困惑度（Perplexity）。

    什么是困惑度（PPL）？
        困惑度是衡量语言模型质量的标准指标，表示模型对测试数据的"困惑程度"。
        - PPL 越低，模型预测能力越强（对数据越不困惑）
        - PPL = exp(平均交叉熵损失)
        - 例如：PPL=10 表示模型平均在 10 个候                                                                                                  
    如何计算？        
        计算预测分布与真实 token 之间的交叉熵损失，然后对所有位置求平均。

    参数：
        model: HuggingFace 因果语言模型（如 GPTNeoX）
        data_chunks: input_ids 张量列表，每个形状为 [1, seq_len]
            例如：[tensor([1, 2048]), tensor([1, 2048]), ...]
            每个 chunk 是一段连续的文本，长度为 max_length
        device: "cuda" 或 "cpu"
        desc: 进度条描述文字

    返回：
        float: 困惑度值（越低越好）
    """
    model.eval()  # 设置为评估模式（关闭 dropout 等随机性）
    total_loss = 0.0  # 累计总损失（所有 token 的交叉熵之和）
    total_tokens = 0  # 累计总 token 数（用于计算平均损失）

    # 遍历所有数据块，逐块计算损失
    for chunk in tqdm(data_chunks, desc=desc):
        input_ids = chunk.to(device)  # 将数据移到指定设备
        seq_len = input_ids.size(1)   # 获取序列长度，例如 2048

        # ====================================================================
        # 前向传播：计算自回归语言建模损失
        # ====================================================================
        # labels=input_ids 表示用自身作为目标（自回归语言建模）
        # 模型会自动计算每个位置预测下一个 token 的交叉熵损失
        #
        # 具体过程：
        # - 位置 0 预测位置 1：loss_0 = CrossEntropy(logits[0]                                                                                                                                                                                                                                          , input_ids[1])
        # - 位置 1 预测位置 2：loss_1 = CrossEntropy(logits[1], input_ids[2])
        # - ...
        # - 位置 n-2 预测位置 n-1：loss_{n-2} = CrossEntropy(logits[n-2], input_ids[n-1])
        # - 最后一个位置没有预测目标，所以总共 seq_len-1 个损失
        #
        # outputs.loss 是这些损失的平均值
        outputs = model(input_ids=input_ids, labels=input_ids)

        # ====================================================================
        # 累计损失和 token 数
        # ====================================================================
        # outputs.loss 是该 chunk 内所有 token 的平均损失
        # 乘以 (seq_len - 1) 还原为总损失
        # 为什么是 seq_len-1？因为第一个 token 没有预测目标（没有前文）
        total_loss += outputs.loss.item() * (seq_len - 1)
        total_tokens += seq_len - 1

    # ====================================================================
    # 计算困惑度
    # ====================================================================
    # 计算所有 token 的平均损失
    avg_loss = total_loss / total_tokens
    # 困惑度 = e^(平均损失)
    # 为什么用 exp？因为交叉熵损失是负对数似然的平均值
    # PPL = exp(-log P(data)) = exp(CrossEntropy)
    ppl = np.exp(avg_loss)
    return ppl


@torch.no_grad()
def measure_generation_time(model, input_ids, gen_length=128, device="cuda", num_runs=3):
    """
    测量推理速度的两个关键指标：TTFT 和 TPOT。

    推理过程分为两个阶段：
        1. Prefill（预填充）阶段：
           - 将整个 prompt 一次性送入模型
           - 计算所有 token 之间的注意力
           - 生成 KV Cache
           - 耗时与 prompt 长度的平方成正比（因为注意力是 O(n²)）

        2. Decode（解码）阶段：
           - 逐个生成新 token
           - 每次只输入 1 个 token，利用 KV Cache 避免重复计算
           - 每步的计算量几乎恒定

    关键指标：
        - TTFT (Time To First Token): 从输入 prompt 到生成第一个 token 的时间
          包含了整个 prefill 阶段的计算。越短表示响应越快。
        - TPOT (Time Per Output Token): 后续每个 token 的平均生成时间
          即 decode 阶段单步耗时。越短表示生成越快。
        - Throughput: 每秒生成的 token 数量 = 1 / TPOT

    参数：
        model: HuggingFace 因果语言模型
        input_ids: prompt 的 token ID 张量，形状 [1, prompt_len]
            例如 [1, 256] 表示 256 个 token 的 prompt
        gen_length: 要生成的 token 数量（默认 128）
        device: "cuda" 或 "cpu"
        num_runs: 运行次数（取平均值，减少随机波动）
            GPU 上的计算时间会有轻微波动，多次运行取平均更准确

    返回：
        dict: 包含 ttft_ms, tpot_ms, throughput_tok_per_sec
    """
    model.eval()
    input_ids = input_ids.to(device)

    ttft_list = []  # 存储每次运行的 TTFT
    tpot_list = []  # 存储每次运行的 TPOT

    for _ in range(num_runs):
        # 如果在 GPU 上，先同步确保之前的操作完成
        # GPU 操作是异步的，torch.cuda.synchronize() 会等待所有 GPU 操作完成
        # 不同步的话，计时可能不准确
        if _is_cuda_device(device):
            torch.cuda.synchronize()

        # ====================================================================
        # === 测量 TTFT：prefill + 第一个 token 的生成时间 ===
        # ====================================================================
        start = time.perf_counter()  # 高精度计时器

        # prefill：将整个 prompt 送入模型，一次性计算所有位置的 attention
        # use_cache=True：启用 KV Cache，返回缓存供后续解码使用
        #
        # 例如 prompt 有 256 个 token：
        # - 模型会计算 256x256 的注意力矩阵
        # - 生成 KV Cache，包含 256 个位置的 Key/Value
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits        # 模型输出的 logit 值，形状 [1, seq_len, vocab_size]
        past_kv = outputs.past_key_values  # KV Cache，包含每一层的 (Key, Value)

        # 取最后一个位置的 logit，贪心解码选择概率最高的 token
        # logits[:, -1, :] 形状: [1, vocab_size]
        # argmax 选择概率最高的 token ID
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # next_token 形状: [1, 1]，表示 1 个生成的 token

        if _is_cuda_device(device):
            torch.cuda.synchronize()  # 等待 GPU 计算完成再计时
        ttft = time.perf_counter() - start  # 计算 TTFT（单位：秒）
        ttft_list.append(ttft)

        # ====================================================================
        # === 测量 TPOT：后续每个 token 的生成时间 ===
        # ====================================================================
        generated_tokens = [next_token]  # 已生成的 token 列表
        start = time.perf_counter()

        for _ in range(gen_length - 1):
            # ---- 自回归解码（每次只输入 1 个 token） ----
            # input_ids=next_token：只输入上一步生成的 token（形状 [1, 1]）
            # past_key_values=past_kv：传入 KV Cache，避免重复计算前面的 token
            #
            # 有了 KV Cache，模型只需计算新 token 与所有历史位置的注意力
            # 计算量从 O(n²) 降低到 O(n)
            outputs = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values  # 更新 KV Cache（追加新 token 的 K/V）
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # 贪心解码
            generated_tokens.append(next_token)

        if _is_cuda_device(device):
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - start
        # TPOT = 总解码时间 / 生成的 token 数
        # 注意是 gen_length-1 因为第一个 token 已经在 TTFT 中计时了
        tpot = decode_time / (gen_length - 1) if gen_length > 1 else 0
        tpot_list.append(tpot)

    # 取多次运行的平均值，减少偶然波动
    avg_ttft = np.mean(ttft_list)
    avg_tpot = np.mean(tpot_list)
    throughput = 1.0 / avg_tpot if avg_tpot > 0 else 0  # 吞吐量 = 1 / TPOT（tok/s）

    return {
        "ttft_ms": avg_ttft * 1000,              # 转换为毫秒（秒 * 1000）
        "tpot_ms": avg_tpot * 1000,              # 转换为毫秒
        "throughput_tok_per_sec": throughput,      # 每秒 token 数
    }


def measure_kv_cache_memory(past_key_values):
    """
    计算 KV Cache 的内存占用。

    什么是 KV Cache？
        在 Transformer 的自注意力机制中，每个 token 会生成三个向量：
        - Query (Q)：用于查询其他位置
        - Key (K)：用于被其他位置查询
        - Value (V)：存储该位置的信息

        在自回归生成时，之前位置的 K 和 V 不会变化，所以可以缓存起来。
        这就是 KV Cache，避免了重复计算，但会占用显存。

    KV Cache 的大小：
        - 与序列长度成正比：seq_len 越长，缓存越大
        - 与层数成正比：每层都有独立的 KV Cache
        - 与 head 数量和 head_dim 成正比

        公式：memory = 2 * num_layers * num_heads * seq_len * head_dim * bytes_per_element
        其中 2 是因为有 Key 和 Value 两个张量

    为什么要测量？
        KV Cache 是长序列推理的内存瓶颈。
        例如：Pythia-70M 在 seq_len=2048 时，KV Cache 约占 6 MB（float16）
        对于更大的模型（如 70B），KV Cache 可能占用几十 GB！

    KV Cache 结构：
        past_key_values 是一个 tuple，每个元素对应一层 Transformer
        每层包含 (key_tensor, value_tensor)
        每个张量形状：[batch, num_heads, seq_len, head_dim]

        例如 Pythia-70M（6 层，8 个 head，head_dim=8）：
        - past_key_values 长度 = 6
        - 每个 key/value 形状: [1, 8, seq_len, 8]

    参数：
        past_key_values: 模型输出的 KV Cache，tuple of (key, value) per layer

    返回：
        dict: 包含 total_elements（总元素数）和 memory_mb（内存占用，单位 MB）
    """
    total_elements = 0  # 累计所有 KV 张量的元素总数

    # 遍历每一层的 KV Cache
    for k, v in _iter_cache_layers(past_key_values):
        # numel() 返回张量中的元素总数
        # 例如：形状 [1, 8, 2048, 8] 的张量有 1*8*2048*8 = 131,072 个元素
        total_elements += k.numel() + v.numel()

    # 计算内存占用
    # 假设使用 float16（半精度），每个元素占 2 字节
    # 如果使用 float32（全精度），每个元素占 4 字节
    memory_bytes = total_elements * 2  # 2 bytes per float16 element
    memory_mb = memory_bytes / (1024 * 1024)  # 转换为 MB（1 MB = 1024*1024 bytes）

    return {
        "total_elements": total_elements,  # 总元素数
        "memory_mb": memory_mb,            # 内存占用（MB）
    }


@torch.no_grad()
def measure_snapkv_generation_time(model, input_ids, compressor, gen_length=128, device="cuda", num_runs=3):
    """Measure SnapKV TTFT, TPOT, and throughput.

    TTFT includes prefill, attention export, and KV compression overhead.
    """
    from src.snapkv import apply_snapkv

    model.eval()
    input_ids = input_ids.to(device)
    ttft_list = []
    tpot_list = []

    for _ in range(num_runs):
        if _is_cuda_device(device):
            torch.cuda.synchronize()

        start = time.perf_counter()
        past_kv, logits = apply_snapkv(model, input_ids, compressor, device=device)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if _is_cuda_device(device):
            torch.cuda.synchronize()
        ttft_list.append(time.perf_counter() - start)

        start = time.perf_counter()
        for _ in range(gen_length - 1):
            outputs = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if _is_cuda_device(device):
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - start
        tpot_list.append(decode_time / (gen_length - 1) if gen_length > 1 else 0)

    avg_ttft = np.mean(ttft_list)
    avg_tpot = np.mean(tpot_list)
    return {
        "ttft_ms": avg_ttft * 1000,
        "tpot_ms": avg_tpot * 1000,
        "throughput_tok_per_sec": 1.0 / avg_tpot if avg_tpot > 0 else 0,
    }


def get_cache_sequence_length(past_key_values):
    """Return the sequence length stored in a KV cache."""
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except TypeError:
            return int(past_key_values.get_seq_length(0))
    for k, _ in _iter_cache_layers(past_key_values):
        return int(k.shape[-2])
    return 0


def estimate_generation_flops(config, prompt_length, gen_length=128, decode_cache_length=None):
    """
    Estimate FLOPs for one prompt prefill plus autoregressive generation.

    This includes Transformer QKV/output projections, attention (QK and AV),
    MLP, and lm_head matmuls. It excludes sampling, top-k indexing, cache
    movement, and other non-matmul overhead.
    """
    hidden_size = int(getattr(config, "hidden_size"))
    num_layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
    intermediate_size = int(getattr(config, "intermediate_size", 4 * hidden_size))
    vocab_size = int(getattr(config, "vocab_size"))
    prompt_length = int(prompt_length)
    gen_length = int(gen_length)
    decode_cache_length = prompt_length if decode_cache_length is None else int(decode_cache_length)

    def layer_flops(seq_len, context_len):
        # Count multiply-add as 2 FLOPs.
        qkv_and_out = 8 * seq_len * hidden_size * hidden_size
        mlp = 4 * seq_len * hidden_size * intermediate_size
        attention = 4 * seq_len * context_len * hidden_size
        return num_layers * (qkv_and_out + mlp + attention)

    prefill_flops = layer_flops(prompt_length, prompt_length) + 2 * prompt_length * hidden_size * vocab_size

    decode_flops = 0
    for step in range(gen_length):
        context_len = decode_cache_length + step + 1
        decode_flops += layer_flops(1, context_len) + 2 * hidden_size * vocab_size

    total_flops = prefill_flops + decode_flops
    return {
        "prompt_length": prompt_length,
        "gen_length": gen_length,
        "decode_cache_length": decode_cache_length,
        "prefill_flops": float(prefill_flops),
        "decode_flops": float(decode_flops),
        "total_flops": float(total_flops),
        "avg_flops_per_output_token": float(total_flops / gen_length) if gen_length else 0.0,
        "total_gflops": float(total_flops / 1e9),
        "avg_gflops_per_output_token": float(total_flops / gen_length / 1e9) if gen_length else 0.0,
    }
