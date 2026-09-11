# 第一周：LLM 推理

- [LLM 推理课件](LLM推理tutorial.pptx)
- [Qwen 推理调试脚本](qwen_inference.py)
- [课后作业](作业.md)

课件以 Qwen3-8B 为例讲解自回归生成、hidden states、Prefill / Decode 与 KV cache。本作业使用较小的 **Qwen2.5-1.5B-Instruct**，可选 **Qwen2.5-3B-Instruct**，便于在 CPU 上完成。它们使用 Qwen2 架构，不要照搬课件 Qwen3 的层数、维度或 Q/K norm 代码；实际参数见脚本的 `model_config`。Qwen2.5 不需要 `enable_thinking=False`。

课件第 46 页提到的 `code/minimal_inference.py`、`collect_trace.py` 等原配套文件不在当前仓库。本目录提供独立的入门作业脚本，课件原文件保留。

## 安装

推荐 Python 3.10–3.12，在本目录创建独立环境：

```bash
python -m venv .venv
```

Windows PowerShell 激活：`.venv\Scripts\Activate.ps1`；Linux/macOS：`source .venv/bin/activate`。以下方案任选其一。

CPU（Windows/Linux）：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

NVIDIA GPU 示例（驱动须支持 CUDA 12.6 对应 wheel）：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

其他系统/驱动组合参见 [PyTorch 安装矩阵](https://pytorch.org/get-started/previous-versions/#v271)。macOS 可安装 `torch==2.7.1` 并使用 CPU 模式。Transformers 固定为课件版本 4.51.3。

## 运行

CPU 入门（最多生成 16 个 token，带调试）：

```bash
python qwen_inference.py --device cpu --max-new-tokens 16 --debug --output results/cpu.json
```

GPU 或 3B 模型：

```bash
python qwen_inference.py --device cuda --debug --output results/gpu.json
python qwen_inference.py --model Qwen/Qwen2.5-3B-Instruct --device cuda --debug --output results/3b.json
```

自定义问题：

```bash
python qwen_inference.py --prompt "请用两句话解释什么是大语言模型。" --max-new-tokens 32 --debug --output results/custom.json
```

不指定设备时自动选择可用的 CUDA，否则用 CPU。CPU 使用 FP32；CUDA 优先 BF16，不支持则用 FP16。1.5B 的 FP32 权重约 6 GB，3B 约 12 GB；加载、激活和 cache 还需要额外内存。内存紧张时选 1.5B、短输入和少量输出。

## 模型下载

推荐先通过 [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct) 下载，再将本地目录传给脚本：

```bash
python -m pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct')"
python qwen_inference.py --model ./models/Qwen2.5-1.5B-Instruct --device cpu --debug
```

ModelScope 仅用于下载，推理仍使用 Transformers。

直接使用模型 ID 运行时，脚本自动从 Hugging Face 下载并缓存模型。也可先下载到指定目录，再从本地加载：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct')"
python qwen_inference.py --model ./models/Qwen2.5-1.5B-Instruct --device cpu --debug
```

3B 模型将 ID 和目录中的 `1.5B` 改为 `3B`。下载需要访问所选模型站点；也可在联网机器下载后复制完整目录，保留 config、tokenizer 和全部权重文件。脚本开头也附有这些下载命令。

## 结果与计时

脚本打印结果并保存 UTF-8 JSON。不同实验使用不同 `--output`；同名文件会覆盖。

| 字段 | 含义 |
| --- | --- |
| `output_text` | 只解码新增 token，去除特殊 token 后的回答 |
| `generation_seconds` | 完整预热一次后，单次 `generate()` 墙钟耗时（秒） |
| `load_seconds` | tokenizer/模型加载时间，首次下载也包含在内 |
| `generated_tokens` | 实际新增 token 数，包括 EOS 等特殊 token，可能小于上限 |
| `tokens_per_second` | 实际新增 token 数 / 生成时间，包含 Prefill 开销 |
| `input_ids` / `input_shape` | 包括聊天模板标记的输入 ID / 形状 |
| `debug` | 额外 forward 的中间变量，仅在 `--debug` 时存在 |

生成计时包含 Prefill 和 Decode，不包含加载、分词、解码文本、调试或打印。CUDA 在计时前后同步，等待异步计算完成。预热执行相同任务一次，正式测量一次，并非多次平均值。脚本使用 greedy decoding、KV cache 与 eager attention，作为教学观察工具，不代表优化后的性能上限。

调试部分对应课件第 13、32–33、35–40 页：

- `hidden_states` 记录状态 0、1、最后一个状态的形状及 `[0,-1,:8]` 数值。0 为 embedding，1 为首个 block 后的状态，最后一个已通过 final RMSNorm；状态总数为 block 数 + 1。
- Prefill 状态形状为 `[B,S,H]`：本例 B=1，S 是含模板的输入 token 数，H 为 hidden size。
- `logits` 是 `[B,S,V]` 的未归一化词表分数，不是概率。`top5_token_ids` 是最后位置分数最高的五个候选。
- Prefill 预测首个生成 token，再将该 token 输入模型，观察 Decode 状态 `[1,1,H]`；若首 token 为 EOS，则跳过 Decode。
- 首层 K/V cache 为 `[B,KV头数,T,head_dim]`。一步 Decode 后 T 从 S 增为 S+1，mask 长度也增长；cache 保存 K/V，不保存所有 hidden states。

只保存索引明确的数值切片，不保存完整张量。实际维度和值以运行结果为准，课件中的 Qwen3 数据不是本作业标准答案。

## CPU 实测记录

2026-09-10 使用 ModelScope 下载的完整 Qwen2.5-1.5B-Instruct 权重验证通过。环境为 Intel Core i9-13900H、CPU FP32、14 个 PyTorch 线程，Python 3.12.14、PyTorch 2.7.1+cpu、Transformers 4.51.3。

- 输入：中国的首都是哪里？请用一句话回答。
- 输出：中国的首都是北京。
- `max_new_tokens=16`，实际新增 6 个 token（含 EOS）。
- 完整预热一次后，单次生成 **1.152 秒**；不含模型加载、分词和调试。
- Prefill hidden states：`[1,39,1536]`；Decode：`[1,1,1536]`。
- 首层 K cache：`[1,2,39,128]` → `[1,2,40,128]`。

这是一次本机实测，不是平均耗时；不同设备和输入的耗时会变化。

## 常见问题

- CUDA 不可用：检查驱动和 PyTorch CUDA wheel，或改用 `--device cpu`。
- 内存不足：使用 1.5B，缩短输入/输出；关闭 `--debug` 可减少调试开销，但作业仍需一次短输入的调试记录。
- 下载失败：检查网络或提前下载完整模型后用本地路径。
- CPU 较慢：先用 `--max-new-tokens 8` 验证；脚本会完整预热一次。作业不按速度排名。

实现参考：[Qwen2.5 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)、[Transformers 输出说明](https://huggingface.co/docs/transformers/v4.51.3/main_classes/output)、[KV cache 说明](https://huggingface.co/docs/transformers/v4.51.3/cache_explanation)。
