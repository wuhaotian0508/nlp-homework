"""
数据加载工具模块
支持的数据集：pg-19（长文本）、wikitext（标准语言模型基准测试集）
"""

import torch
from datasets import load_dataset  # HuggingFace 数据集库，用于下载和加载公开数据集
from transformers import AutoTokenizer  # HuggingFace 分词器


PG19_FALLBACK_DATASET = "emozilla/pg19-test"


def safe_print(text):
    """在 Windows 非 UTF-8 终端中尽量安全地输出文本。"""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        print(safe_text)


def load_wikitext(tokenizer, split="test", max_length=2048):
    """
    加载并分词 WikiText-2 数据集。

    什么是 WikiText？
        WikiText 是一个标准的语言模型评估数据集，来自维基百科文章。
        - WikiText-2: 约 2M tokens，适合快速评估
        - WikiText-103: 约 100M tokens，更大规模
        这里使用 WikiText-2 的原始版本（未经过预处理）

    为什么用 WikiText？
        - 标准基准：几乎所有语言模型论文都会报告 WikiText PPL
        - 质量高：维基百科文章语法规范，覆盖多种主题
        - 适中长度：不像 PG-19 那样超长，适合快速实验

    参数：
        tokenizer: HuggingFace 分词器
            例如 GPTNeoXTokenizerFast，将文本转换为 token ID
        split: 数据集划分（"test"测试集, "train"训练集, "validation"验证集）
            评估 PPL 通常使用 test 集
        max_length: 每个分块的最大序列长度
            例如 2048，将长文本切分成 2048 token 的块

    返回：
        list[Tensor]: 分词后的 input_ids 张量列表，每个形状为 [1, max_length]
            例如：[tensor([[1, 2, 3, ..., 2048]]), tensor([[...]])]
    """
    # ====================================================================
    # 第 1 步：从 HuggingFace Hub 加载 WikiText-2 原始文本版本
    # ====================================================================
    # load_dataset 会自动下载数据集到本地缓存（~/.cache/huggingface/datasets/）
    # "wikitext-2-raw-v1" 表示未经过预处理的原始版本
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # ====================================================================
    # 第 2 步：将数据集中所有非空文本拼接成一个长字符串
    # ====================================================================
    # dataset 是一个字典列表，每个元素有 "text" 字段
    # 例如：[{"text": "Article 1..."}, {"text": ""}, {"text": "Article 2..."}]
    # 过滤掉空文本，用双换行符分隔不同文章
    text = "\n\n".join([item["text"] for item in dataset if item["text"].strip()])

    # ====================================================================
    # 第 3 步：使用分词器将文本转换为 token ID 序列
    # ====================================================================
    # tokenizer(text) 会将文本分词并转换为 ID
    # return_tensors="pt" 表示返回 PyTorch 张量
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]  # 形状 [total_tokens]
    # 例如：tensor([1, 15, 234, 5678, ...])，长度可能是几十万

    # ====================================================================
    # 第 4 步：将长序列切分为固定长度的块
    # ====================================================================
    # 为什么要切分？
    # - 内存限制：一次性处理几十万 token 会 OOM
    # - 并行评估：每个块可以独立计算 PPL，最后平均
    # - 模拟真实场景：实际应用中输入长度有限
    chunks = []
    for i in range(0, len(input_ids) - max_length, max_length):
        # 切片 [i : i+max_length] 取出一个块
        # unsqueeze(0) 升维为 [1, max_length]，因为模型需要 batch 维度
        chunk = input_ids[i : i + max_length].unsqueeze(0)
        chunks.append(chunk)

    # 打印加载信息
    safe_print(f"[WikiText] 加载了 {len(chunks)} 个数据块，每块 {max_length} 个 token")
    # 例如：[WikiText] 加载了 50 个数据块，每块 2048 个 token
    return chunks


def load_pg19(tokenizer, split="test", max_length=2048, num_samples=1):
    """
    加载并分词 PG-19 数据集（长篇文本，如完整书籍）。

    什么是 PG-19？
        PG-19 是一个超长文本数据集，来自古腾堡计划（Project Gutenberg）的书籍。
        - 包含约 28,000 本公版书籍
        - 每个样本是一整本书，长度可达数十万 token
        - 特别适合测试长序列场景下的模型表现

    为什么用 PG-19？
        - 超长序列：一本书的文本长度远超普通数据集
        - 测试 KV Cache 压缩效果：长序列下 KV Cache 更大，压缩效果更明显
        - 作业要求：在 pg-19 上测试 PPL，可取单一 sample 测试

    为什么用流式加载？
        PG-19 完整数据集很大（几十 GB），一次性下载会占用大量磁盘和内存。
        streaming=True 让数据按需加载，只下载实际使用的样本。

    参数：
        tokenizer: HuggingFace 分词器
        split: 数据集划分
        max_length: 最大序列长度（截断用）
            例如 2048，将书籍文本截断到 2048 个 token
        num_samples: 使用的样本数量（按作业要求默认 1 个）
            "可取单一 sample 进行测试即可"

    返回：
        list[Tensor]: 分词后的 input_ids 张量列表，每个形状为 [1, <=max_length]
            每个元素对应一本书（截断后）
    """
    # ====================================================================
    # 第 1 步：使用流式加载 PG-19 数据集
    # ====================================================================
    # 新版 datasets 已不再支持部分脚本式数据集（如原始 pg19）。
    # 因此先尝试官方名称；若失败，则回退到一个可直接从 Hub 读取的镜像数据集。
    dataset_name = "pg19"
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        dataset_name = PG19_FALLBACK_DATASET
        safe_print(
            "[PG-19] 检测到当前 datasets 版本不再支持原始 pg19 脚本，"
            f"自动回退到 {dataset_name}。"
        )
        dataset = load_dataset(dataset_name, split=split, streaming=True)

    # ====================================================================
    # 第 2 步：逐样本分词和截断
    # ====================================================================
    chunks = []
    for idx, item in enumerate(dataset):
        if idx >= num_samples:  # 只取指定数量的样本
            break

        text = item["text"]  # 获取书籍全文
        # 分词并截断到 max_length，防止内存溢出
        # truncation=True：如果文本超过 max_length，自动截断
        # 例如：一本书有 100,000 个 token，截断后只取前 2048 个
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        chunks.append(encodings.input_ids)  # 形状 [1, seq_len]，seq_len <= max_length

    # 打印加载信息
    safe_print(
        f"[PG-19] 从 {dataset_name} 加载了 {len(chunks)} 个样本，每个最多 {max_length} 个 token"
    )
    # 例如：[PG-19] 加载了 1 个样本，每个最多 2048 个 token
    return chunks


def get_dataset(name, tokenizer, max_length=2048, **kwargs):
    """
    统一的数据集加载接口，根据名称选择对应的加载函数。

    参数：
        name: 数据集名称，"wikitext" 或 "pg19"
        tokenizer: HuggingFace 分词器
        max_length: 最大序列长度
    """
    if name == "wikitext":
        return load_wikitext(tokenizer, max_length=max_length, **kwargs)
    elif name == "pg19":
        return load_pg19(tokenizer, max_length=max_length, **kwargs)
    else:
        raise ValueError(f"未知数据集: {name}，请选择 ['wikitext', 'pg19']")



# # 选一个你感兴趣的模型名称
# model_id = "gpt2" 

# # 加载分词器
# tokenizer = AutoTokenizer.from_pretrained(model_id)

if __name__ == "__main__":
    # 1. 初始化分词器
    # 提醒：第一次运行会联网下载几百 KB 的分词配置文件
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 如果是 PG-19 这种需要截断的数据集，必须指定 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 预览 WikiText (看它的短文本块结构)
    safe_print("\n" + "="*30 + " WikiText 预览 " + "="*30)
    wiki_chunks = load_wikitext(tokenizer, max_length=128) # 为了快速查看，设短一点
    if wiki_chunks:
        sample_text = tokenizer.decode(wiki_chunks[0][0])
        safe_print(f"第一个数据块的内容：\n{sample_text[:500]}...") 

    # 3. 预览 PG-19 (看它的长文本书籍结构)
    safe_print("\n" + "="*30 + " PG-19 预览 " + "="*30)
    # num_samples=1 对应你代码中的流式加载，只取第一本书
    pg_chunks = load_pg19(tokenizer, max_length=500, num_samples=1)
    if pg_chunks:
        book_text = tokenizer.decode(pg_chunks[0][0])
        safe_print(f"第一本书的开头：\n{book_text[:500]}...")