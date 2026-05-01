# 个人部分代码说明文档

## 📂 项目结构

```
efficient-inference/
├── src/
│   ├── data.py          # 数据集加载
│   ├── evaluate.py      # 评估工具（PPL、速度测试）
│   └── snapkv.py        # SnapKV 优化实现
├── scripts/
│   ├── eval_baseline.py # Baseline 评估脚本
│   └── eval_snapkv.py   # SnapKV 评估脚本
└── results/             # 结果保存目录
```

---

## 🔍 核心代码位置详解

### 1. 模型下载和导入

**位置：**
- `scripts/eval_baseline.py:39-43`
- `scripts/eval_snapkv.py:115-119`

**代码：**
```python
# 从 HuggingFace 自动下载并加载
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,  # 默认是 "EleutherAI/pythia-70m"
    torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
).to(args.device)
```

**说明：**
- 模型会自动从 HuggingFace Hub 下载到本地缓存（`~/.cache/huggingface/`）
- 使用 Pythia-70M 模型（70M 参数）
- GPU 上使用 float16 半精度加速

---

### 2. 模型优化实现

#### 核心优化算法
**位置：** `src/snapkv.py:41-107` 的 `SnapKVCompressor.compress()`

**优化原理：**
1. **第 61-67 行**：提取观察窗口的注意力权重
2. **第 72 行**：计算每个 KV 位置的重要性分数
3. **第 76-80 行**：应用平均池化平滑重要性分数
4. **第 83-85 行**：选择 top-K 个最重要的位置
5. **第 96-106 行**：收集选中的 KV + 拼接观察窗口

#### 优化应用流程
**位置：** `src/snapkv.py:110-157` 的 `apply_snapkv()`

**流程：**
```python
# 1. 前向传播，获取完整的 KV Cache 和注意力权重（第 134-139 行）
outputs = model(
    input_ids=input_ids,
    use_cache=True,
    output_attentions=True,  # 输出注意力权重
)

# 2. 逐层压缩 KV Cache（第 146-154 行）
for layer_idx in range(len(past_kv)):
    key_states = past_kv[layer_idx][0]
    value_states = past_kv[layer_idx][1]
    attn_weights = attentions[layer_idx]

    # 应用 SnapKV 压缩
    comp_k, comp_v = compressor.compress(key_states, value_states, attn_weights)
    compressed_kv.append((comp_k, comp_v))
```

---

### 3. PPL（困惑度）测试

#### Baseline PPL 测试
**位置：** `src/evaluate.py:12-50` 的 `evaluate_ppl()`

**调用位置：** `scripts/eval_baseline.py:53`

**实现：**
```python
# 使用完整 KV Cache 计算困惑度
for chunk in data_chunks:
    outputs = model(input_ids=input_ids, labels=input_ids)
    total_loss += outputs.loss.item() * (seq_len - 1)
    total_tokens += seq_len - 1

avg_loss = total_loss / total_tokens
ppl = np.exp(avg_loss)  # 困惑度 = e^(平均损失)
```

#### SnapKV PPL 测试
**位置：** `scripts/eval_snapkv.py:23-85` 的 `evaluate_ppl_with_snapkv()`

**实现：**
```python
# 1. 将序列分成前半段和后半段（第 51-54 行）
split_pos = seq_len // 2
prompt_ids = input_ids[:, :split_pos]   # 前半段：压缩 KV
eval_ids = input_ids[:, split_pos:]     # 后半段：评估

# 2. 对前半段压缩 KV Cache（第 57 行）
compressed_kv, _ = apply_snapkv(model, prompt_ids, compressor, device=device)

# 3. 使用压缩后的 KV 在后半段逐 token 计算损失（第 65-77 行）
for t in range(eval_ids.size(1) - 1):
    outputs = model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
    loss = torch.nn.functional.cross_entropy(logits, target_token.squeeze(1))
    loss_sum += loss.item()
```

---

### 4. 加速测试

**位置：** `src/evaluate.py:54-130` 的 `measure_generation_time()`

**调用位置：**
- Baseline: `scripts/eval_baseline.py:60`
- SnapKV: `scripts/eval_snapkv.py:143`

**测量指标：**

#### TTFT (Time To First Token)
**位置：** 第 86-100 行

```python
# 测量 prefill + 第一个 token 的生成时间
start = time.perf_counter()
outputs = model(input_ids=input_ids, use_cache=True)
next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
ttft = time.perf_counter() - start
```

#### TPOT (Time Per Output Token)
**位置：** 第 104-119 行

```python
# 测量后续每个 token 的平均生成时间
start = time.perf_counter()
for _ in range(gen_length - 1):
    outputs = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
decode_time = time.perf_counter() - start
tpot = decode_time / (gen_length - 1)
```

#### Throughput（吞吐量）
**位置：** 第 124 行

```python
throughput = 1.0 / avg_tpot  # 每秒生成的 token 数
```

---

### 5. 数据集加载

**位置：** `src/data.py`

#### WikiText 数据集
**函数：** `load_wikitext()` (第 11-41 行)

```python
# 从 HuggingFace Hub 自动下载
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

# 处理流程：
# 1. 拼接所有文本（第 27 行）
text = "\n\n".join([item["text"] for item in dataset if item["text"].strip()])

# 2. 分词（第 30 行）
encodings = tokenizer(text, return_tensors="pt")

# 3. 切分成固定长度的块（第 36-38 行）
for i in range(0, len(input_ids) - max_length, max_length):
    chunk = input_ids[i : i + max_length].unsqueeze(0)
    chunks.append(chunk)
```

#### PG-19 数据集
**函数：** `load_pg19()` (第 44-72 行)

```python
# 流式加载（避免一次性下载整个数据集）
dataset = load_dataset("pg19", split=split, streaming=True)

# 只取指定数量的样本（默认 1 个）
for idx, item in enumerate(dataset):
    if idx >= num_samples:
        break
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    chunks.append(encodings.input_ids)
```

#### 统一接口
**函数：** `get_dataset()` (第 75-89 行)

**调用位置：**
- Baseline: `scripts/eval_baseline.py:49`
- SnapKV: `scripts/eval_snapkv.py:133`

```python
data_chunks = get_dataset(args.dataset, tokenizer, max_length=args.max_length)
```

---

## 🎯 数据集说明

| 数据集 | 来源 | 用途 | 特点 |
|--------|------|------|------|
| **WikiText-2** | HuggingFace `wikitext` | 标准语言模型基准测试 | 中等长度文本，多个块 |
| **PG-19** | HuggingFace `pg19` | 长文本测试 | 完整书籍，超长序列 |

**缓存位置：** `~/.cache/huggingface/datasets/`

---

## 🔄 完整执行流程

```
1. 运行脚本 (eval_baseline.py 或 eval_snapkv.py)
   ↓
2. 下载/加载模型 (from_pretrained)
   ↓
3. 加载数据集 (get_dataset → load_wikitext/load_pg19)
   ↓
4. PPL 测试
   - Baseline: evaluate_ppl()
   - SnapKV: evaluate_ppl_with_snapkv() → apply_snapkv() → compress()
   ↓
5. 速度测试 (measure_generation_time)
   - 测量 TTFT、TPOT、Throughput
   ↓
6. 内存测试 (measure_kv_cache_memory)
   ↓
7. 保存结果到 JSON (results/baseline.json 或 results/snapkv.json)
```

---

## 📊 评估指标

### 困惑度 (PPL)
- **定义：** PPL = exp(平均交叉熵损失)
- **意义：** 越低越好，表示模型预测能力越强
- **计算：** 在测试集上逐 token 计算预测损失

### 速度指标
- **TTFT：** 首个 token 生成时间（包含 prefill）
- **TPOT：** 每个 token 平均生成时间（decode 阶段）
- **Throughput：** 每秒生成的 token 数量

### 内存指标
- **KV Cache Memory：** KV Cache 占用的显存大小
- **Compression Ratio：** 压缩后内存占用比例

---

## 🚀 运行命令

### Baseline 评估
```bash
python scripts/eval_baseline.py \
    --model EleutherAI/pythia-70m \
    --dataset wikitext \
    --max_length 2048 \
    --device cuda
```

### SnapKV 评估
```bash
python scripts/eval_snapkv.py \
    --model EleutherAI/pythia-70m \
    --dataset wikitext \
    --max_length 2048 \
    --max_capacity 128 \
    --observation_window 32 \
    --kernel_size 5 \
    --device cuda
```

---

## 📝 关键参数说明

### SnapKV 参数
- `--max_capacity`：每个 head 最多保留的 KV 位置数（默认 128）
- `--observation_window`：观察窗口大小（默认 32）
- `--kernel_size`：平滑池化核大小（默认 5）

### 通用参数
- `--model`：模型名称（默认 EleutherAI/pythia-70m）
- `--dataset`：数据集选择（wikitext 或 pg19）
- `--max_length`：最大序列长度（默认 2048）
- `--device`：运行设备（cuda 或 cpu）
