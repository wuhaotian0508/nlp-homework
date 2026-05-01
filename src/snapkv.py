"""
SnapKV: 基于注意力加权选择的高效 KV Cache 压缩算法

核心思想：
  在 prefill（预填充）阶段后，利用"观察窗口"（最后几个 token）的注意力分数
  来识别哪些 KV 位置最重要。只保留 top-k 个重要位置 + 最近的窗口。

参考论文：
  SnapKV: LLM Knows What You are Looking for Before Generation
  https://arxiv.org/abs/2405.20233
"""

import torch
import torch.nn.functional as F


class SnapKVCompressor:
    """
    通过分析观察窗口的注意力模式来压缩 KV Cache。

    工作原理：
    1. 使用最后 N 个 token（观察窗口）的注意力分布作为"未来关注模式"的代理
    2. 统计每个历史位置被观察窗口关注的总分数（重要性投票）
    3. 每个 attention head 独立选择最重要的 K 个位置
    4. 保留 top-K 重要位置 + 最近的观察窗口
    """

    def __init__(self, kernel_size=5, observation_window=32, max_capacity=128):
        """
        初始化 SnapKV 压缩器。

        参数：
            kernel_size: 平滑注意力分数的池化核大小（捕捉局部聚集模式）
            observation_window: 观察窗口大小（最后 N 个 token 作为查询）
            max_capacity: 每个 head 最多保留的 KV 位置数（不含观察窗口）
        """
        self.kernel_size = kernel_size
        self.observation_window = observation_window
        self.max_capacity = max_capacity

    def compress(self, key_states, value_states, attention_weights):
        """
        对单层的 KV Cache 进行压缩。

        参数：
            key_states:   [batch, num_heads, seq_len, head_dim]
                当前层的 Key 张量，存储了所有位置的 key 向量
            value_states: [batch, num_heads, seq_len, head_dim]
                当前层的 Value 张量，存储了所有位置的 value 向量
            attention_weights: [batch, num_heads, seq_len, seq_len]
                完整的注意力矩阵（来自 prefill 阶段）
                attention_weights[b, h, i, j] 表示第 b 个样本、第 h 个 head、
                第 i 个 query 位置对第 j 个 key 位置的注意力权重

        返回：
            compressed_key:   [batch, num_heads, compressed_len, head_dim]
                压缩后的 Key，长度 = max_capacity + observation_window
            compressed_value: [batch, num_heads, compressed_len, head_dim]
                压缩后的 Value，长度 = max_capacity + observation_window
        """
        # 获取张量的维度信息
        bsz, num_heads, seq_len, head_dim = key_states.shape

        # 【早停检查】如果序列足够短，无需压缩，直接返回原始 KV
        # 例如：seq_len=150, max_capacity=128, observation_window=32
        # 150 <= 160，无需压缩
        if seq_len <= self.max_capacity + self.observation_window:
            return key_states, value_states

        # ========================================================================
        # === 步骤 1: 提取观察窗口的注意力权重 ===
        # ========================================================================
        # 核心思想：使用最后 N 个 token（观察窗口）的注意力分布来判断
        # 哪些历史位置是重要的
        #
        # 例如：seq_len=200, observation_window=32
        # - 观察窗口：位置 168-199（最后 32 个 token）
        # - 前缀部分：位置 0-167（前 168 个 token）
        #
        # attention_weights 形状: [bsz, heads, query_len, kv_len]
        # 其中 query_len = kv_len = seq_len（因为是完整的自注意力矩阵）
        #
        # 我们只关心：观察窗口（query）对前缀部分（key）的注意力
        # 不包括观察窗口内部的自注意力（因为这些位置会被完整保留）
        obs_attn = attention_weights[
            :, :, -self.observation_window :, : -self.observation_window
        ]
        # obs_attn 形状: [bsz, heads, obs_window, prefix_len]
        # 例如：[1, 8, 32, 168] 表示 8 个 head，32 个观察位置，168 个前缀位置

        # ========================================================================
        # === 步骤 2: 计算每个前缀位置的重要性分数 ===
        # ========================================================================
        # 对观察窗口的所有查询求和，得到每个历史位置的总关注度
        # 直觉：如果一个历史位置被观察窗口的多个 token 关注，说明它很重要
        #
        # 例如：obs_attn[0, 0, :, 50] 是第 0 个 head 的 32 个观察位置
        # 对第 50 个前缀位置的注意力权重
        # importance[0, 0, 50] = sum(obs_attn[0, 0, :, 50]) 表示总关注度
        importance = obs_attn.sum(dim=2)  # [bsz, heads, prefix_len]
        # 例如：[1, 8, 168] 表示 8 个 head，每个 head 对 168 个前缀位置的重要性评分

        # ========================================================================
        # === 步骤 3: 应用平均池化平滑重要性分数 ===
        # ========================================================================
        # 为什么要平滑？
        # - 注意力权重可能有噪声，某个位置偶然获得高分
        # - 平滑可以捕捉"局部聚集"的注意力模式
        # - 例如：如果位置 50-54 都被关注，平滑后这个区域的重要性会更高
        #
        # 使用 1D 平均池化（avg_pool1d）在序列维度上滑动窗口
        if self.kernel_size > 1:
            # padding = kernel_size // 2 保证输出长度不变
            # 例如：kernel_size=5, padding=2
            # 对于位置 i，取 [i-2, i-1, i, i+1, i+2] 的平均值
            padding = self.kernel_size // 2
            importance = F.avg_pool1d(
                importance, kernel_size=self.kernel_size, padding=padding, stride=1
            )
            # importance 形状仍然是 [bsz, heads, prefix_len]

        # ========================================================================
        # === 步骤 4: 选择 top-K 个最重要的位置（每个 head 独立选择）===
        # ========================================================================
        # 每个 attention head 可能关注不同的模式，所以独立选择
        # 例如：head 0 可能关注主语，head 1 可能关注动词
        #
        # select_count = min(max_capacity, prefix_len)
        # 例如：max_capacity=128, prefix_len=168, select_count=128
        select_count = min(self.max_capacity, importance.shape[-1])

        # topk 返回最大的 K 个值及其索引
        # indices 形状: [bsz, heads, select_count]
        # 例如：[1, 8, 128] 表示 8 个 head，每个选择 128 个位置
        _, indices = importance.topk(select_count, dim=-1)

        # 【重要】排序索引以保持原始顺序
        # 为什么要排序？因为 topk 返回的索引是按重要性排序的，不是按位置排序的
        # 例如：topk 可能返回 [100, 50, 150, 20, ...]
        # 排序后变成 [20, 50, 100, 150, ...] 保持时间顺序
        # 这对于保持因果关系很重要！
        indices, _ = indices.sort(dim=-1)

        # ========================================================================
        # === 步骤 5: 收集选中的 KV + 拼接观察窗口 ===
        # ========================================================================
        # 使用 torch.gather 根据索引收集选中的 KV 位置
        #
        # 扩展 indices 以匹配 head_dim 维度
        # indices 形状: [bsz, heads, select_count]
        # indices_k 形状: [bsz, heads, select_count, head_dim]
        # 例如：[1, 8, 128] -> [1, 8, 128, 64]（假设 head_dim=64）
        indices_k = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        # 分离出前缀部分（不含观察窗口）
        # 例如：seq_len=200, observation_window=32
        # prefix_key 是位置 0-167 的 key
        prefix_key = key_states[:, :, : -self.observation_window, :]
        prefix_value = value_states[:, :, : -self.observation_window, :]

        # 根据 indices_k 收集选中的 KV 位置
        # gather(dim=2, index=indices_k) 在序列维度（dim=2）上按索引收集
        # selected_key 形状: [bsz, heads, select_count, head_dim]
        # 例如：[1, 8, 128, 64] 表示选中了 128 个重要位置
        selected_key = prefix_key.gather(2, indices_k)
        selected_value = prefix_value.gather(2, indices_k)

        # 观察窗口始终保留（最近的 token 通常很重要）
        # recent_key 是位置 168-199 的 key（最后 32 个）
        recent_key = key_states[:, :, -self.observation_window :, :]
        recent_value = value_states[:, :, -self.observation_window :, :]

        # 拼接：[top-K 重要位置] + [最近窗口]
        # compressed_key 形状: [bsz, heads, select_count + obs_window, head_dim]
        # 例如：[1, 8, 128+32, 64] = [1, 8, 160, 64]
        # 原始长度 200 压缩到 160，压缩率 80%
        compressed_key = torch.cat([selected_key, recent_key], dim=2)
        compressed_value = torch.cat([selected_value, recent_value], dim=2)

        return compressed_key, compressed_value


def apply_snapkv(model, input_ids, compressor, device="cuda"):
    """
    完整的 SnapKV 压缩流程：先 prefill，再压缩 KV Cache。

    这是 SnapKV 的主入口函数，完成从输入到压缩后 KV Cache 的全流程。

    什么是 prefill？
        在 Transformer 推理中，prefill 是第一阶段：将整个 prompt 一次性送入模型，
        计算所有位置之间的注意力，并生成 KV Cache。
        第二阶段 decode 则是逐 token 生成，利用 KV Cache 避免重复计算。

    什么是 KV Cache？
        在自注意力机制中，每个 token 会生成 Key 和 Value 向量。
        在 decode 阶段，之前位置的 Key/Value 不会变化，所以缓存起来避免重复计算。
        KV Cache 的大小与序列长度成正比，是内存瓶颈之一。

    SnapKV 做了什么？
        在 prefill 完成后，分析注意力权重，只保留最重要的 KV 位置，
        从而减小 KV Cache 的大小，加速后续的 decode 阶段。

    流程：
    1. 将 prompt 送入模型前向传播，获取完整的注意力权重和 KV Cache
    2. 对每一层的 KV Cache 应用 SnapKV 压缩
    3. 返回压缩后的 KV Cache，供后续解码使用

    参数：
        model: HuggingFace 因果语言模型（如 GPTNeoXForCausalLM）
            Pythia-70M 有 6 层 Transformer，每层 8 个 attention head
        input_ids: prompt 的 token ID，形状 [1, seq_len]
            例如 [1, 256] 表示 1 个样本，256 个 token
        compressor: SnapKVCompressor 实例
            包含 max_capacity、observation_window、kernel_size 等压缩参数
        device: "cuda" 或 "cpu"

    返回：
        compressed_past_key_values: 压缩后的 KV Cache
            格式: tuple of (Key, Value) per layer
            每个 Key/Value 形状: [bsz, heads, compressed_len, head_dim]
        logits: prefill 阶段的模型输出
            形状 [1, seq_len, vocab_size]，可用于计算 PPL 或取 argmax 得到预测 token
    """
    model.eval()  # 设置为评估模式（关闭 dropout 等随机性，保证结果可复现）
    input_ids = input_ids.to(device)  # 将输入移到指定设备（GPU 或 CPU）

    # ====================================================================
    # 第 1 步：Prefill 前向传播，获取完整的注意力权重和 KV Cache
    # ====================================================================
    # output_attentions=True: 让模型返回每一层的注意力矩阵
    # 正常推理不需要这个（会额外消耗显存），但 SnapKV 需要分析注意力模式
    with torch.no_grad():  # 推理阶段不需要梯度，节省显存
        outputs = model(
            input_ids=input_ids,
            use_cache=True,          # 启用 KV Cache，模型会返回 past_key_values
            output_attentions=True,  # 输出注意力权重（SnapKV 需要用到）
        )

    # 解析模型输出
    logits = outputs.logits
    # logits 形状: [1, seq_len, vocab_size]
    # 例如: [1, 256, 50304]，其中 50304 是 Pythia 的词表大小
    # logits[0, i, :] 是模型在位置 i 预测下一个 token 的 logit 分布

    past_kv = outputs.past_key_values
    # past_kv 可能是 tuple 或 DynamicCache 对象（transformers >= 4.36）
    # 如果是 DynamicCache，需要转换为 tuple 格式
    if hasattr(past_kv, 'layers'):
        # DynamicCache: 通过 layers[i].keys 和 layers[i].values 访问
        past_kv = tuple((layer.keys, layer.values) for layer in past_kv.layers)
    # past_kv 是一个 tuple，长度 = 层数（Pythia-70M 有 6 层）
    # past_kv[layer] = (key_tensor, value_tensor)
    # 每个张量形状: [bsz, num_heads, seq_len, head_dim]
    # Pythia-70M: [1, 8, 256, 8]（8 个 head，head_dim=8，因为 hidden=64/8=8）

    attentions = outputs.attentions
    # attentions 也是一个 tuple，长度 = 层数
    # attentions[layer] 形状: [bsz, num_heads, seq_len, seq_len]
    # 例如: [1, 8, 256, 256] 表示 256x256 的完整注意力矩阵

    # ====================================================================
    # 第 2 步：逐层压缩 KV Cache
    # ====================================================================
    # 每一层的注意力模式不同，所以需要独立压缩
    compressed_kv = []
    for layer_idx in range(len(past_kv)):
        key_states = past_kv[layer_idx][0]    # 该层的 Key，[bsz, heads, seq_len, head_dim]
        value_states = past_kv[layer_idx][1]  # 该层的 Value，[bsz, heads, seq_len, head_dim]
        attn_weights = attentions[layer_idx]  # 该层的注意力权重，[bsz, heads, seq_len, seq_len]

        # 应用 SnapKV 压缩：根据注意力模式选择重要的 KV 位置
        # 压缩后 seq_len 维度会从原始长度缩减到 max_capacity + observation_window
        comp_k, comp_v = compressor.compress(key_states, value_states, attn_weights)
        compressed_kv.append((comp_k, comp_v))

    # 转换为与输入相同的格式
    # 如果输入是 DynamicCache，输出也应该是 DynamicCache
    if hasattr(outputs.past_key_values, 'layers'):
        from transformers import DynamicCache
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(compressed_kv):
            cache.update(k, v, layer_idx)
        compressed_past_key_values = cache
    else:
        # 否则返回 tuple 格式
        compressed_past_key_values = tuple(compressed_kv)

    return compressed_past_key_values, logits
