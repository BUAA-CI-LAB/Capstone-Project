# 第六周第 2 次课：DGX Spark 本地部署与 vLLM

本次实验在 DGX Spark 上用 vLLM 部署 Qwen3-1.7B，得到一个 OpenAI 兼容的本地服务。先从命令行和 Python 调用它，再修改启动参数，观察内存、上下文长度和速度的变化。第 7 周把 Agent 迁移到 DGX Spark 时，用的就是这样部署的服务。

- [DGX Spark 登录手册](../dgx-spark-login.md)
- [chat_demo.py](chat_demo.py)：调用服务，统计 TTFT 和生成速度
- [model_info.py](model_info.py)：读取模型配置，估算权重和 KV cache 大小
- [requirements.txt](requirements.txt)：vLLM 与 OpenAI SDK

文中的输出和数字都是 2026-10-14 在 Spark 上实测的结果（vLLM 0.31.0，机器上只运行这一个服务）。多组同时使用时，数字会有波动。

## 0. 准备

1. 确认能按 [DGX Spark 登录手册](../dgx-spark-login.md) 登录 Spark。
2. 在自己电脑上进入课程仓库的 `2026-capstone` 目录，把本目录传到 Spark：

   ```bash
   scp -r week06-2-dgx-local-serving spark:~/week6
   ```

   没有配置 `Host spark` 简称时，写完整地址：`scp -P 2222 -r week06-2-dgx-local-serving 你的用户名@10.134.43.123:~/week6`。

3. 本组端口是 **8100 + 组号**（第 3 组为 8103）。所有组共用一台机器，端口不能重复。
4. 准备两个终端（或 tmux 的两个窗口）：终端 1 运行服务，终端 2 发请求。

## 1. 认识这台机器

```bash
uname -m       # aarch64：ARM 架构
nproc          # CPU 核数
free -h        # 内存总量和剩余，GPU 和 CPU 共用
nvidia-smi     # GPU 型号、驱动版本、CUDA 版本
```

| 项目 | DGX Spark | 对部署的影响 |
| --- | --- | --- |
| GPU | NVIDIA GB10（Blackwell） | 需要 CUDA 13 版本的 PyTorch 和 vLLM |
| 内存 | 128 GB 统一内存，系统可见 121.7 GiB | 没有独立显存，GPU、CPU 和所有组共用这块内存。`nvidia-smi` 的显存一栏显示 N/A，看内存用 `free -h` |
| 内存带宽 | 273 GB/s | 生成每个 token 都要把全部权重读一遍，带宽决定单个请求的生成速度上限 |
| CPU | 20 核 ARM | 只能安装 aarch64 版本的软件包 |

以 Qwen3-1.7B 为例：加载后的权重约 3.5 GB，单个请求的生成速度上限约 273 ÷ 3.5 ≈ 78 token/s，实测约 50 token/s。Qwen3-8B 的权重有 16.4 GB，上限只有约 17 token/s。

## 2. 配置环境

```bash
python3 -m venv ~/venvs/vllm
source ~/venvs/vllm/bin/activate
pip install -U pip uv -i https://mirrors.bfsu.edu.cn/pypi/web/simple
uv pip install vllm==0.31.0 openai -i https://mirrors.bfsu.edu.cn/pypi/web/simple
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
```

最后一行应输出 `0.31.0 2.13.0+cu130 True`。

- `uv` 是更快的 pip，用法基本相同。实测安装约 1 分钟，环境占 7.6 GB。**同组共用一个账号，一组装一份即可。**
- 镜像用北外（BFSU）：2026-10-14 在 Spark 上实测，北外约 35 MB/s，清华约 3 MB/s。北外也慢时，可换成华为云 `https://repo.huaweicloud.com/repository/pypi/simple`。

终端 1 和终端 2 都要先执行下面三行，以后每打开一个新终端也一样：

```bash
source ~/venvs/vllm/bin/activate
export PORT=8103          # 换成 8100 + 组号
export HF_HUB_OFFLINE=1   # Spark 连不上 Hugging Face，不让 vLLM 尝试联网
```

## 3. 看模型参数，估算内存

课程组已下载的模型，所有人可读：

| 模型 | 路径 | 说明 |
| --- | --- | --- |
| Qwen3-1.7B | `/srv/spark/models/Qwen3-1.7B` | 本次实验使用 |
| Qwen3-8B | `/srv/spark/models/Qwen3-8B` | 效果更好，但单个服务需要 `--gpu-memory-utilization 0.2` 左右，课后错峰使用 |

```bash
cd ~/week6
python model_info.py /srv/spark/models/Qwen3-1.7B
```

```text
model_type                qwen3
num_hidden_layers         28
hidden_size               2048
num_attention_heads       16
num_key_value_heads       8
head_dim                  128
vocab_size                151936
tie_word_embeddings       True
max_position_embeddings   40960
torch_dtype               bfloat16

权重文件合计：3.78 GiB
  其中 lm_head 与 embedding 共用的 0.58 GiB 在文件里存了两份，加载后约 3.20 GiB
每个 token 的 KV cache：2 × 28 层 × 8 个 KV 头 × 128 维 × 2 字节 = 112 KiB
一个 40960 token 的请求需要 KV cache：4.38 GiB
```

- 权重约等于参数量 × 2 字节（BF16）。Qwen3-1.7B 的输入 embedding 和输出层 lm_head 共用一份权重（`tie_word_embeddings`），所以加载后比文件小。
- 生成时，每一层都要为每个 token 保存一份 K 和一份 V。Qwen3 使用 GQA，16 个 Q 头共享 8 个 KV 头，KV cache 只有不共享时的一半。
- 模型默认支持 40960 token 的上下文，一个满长请求就要 4.38 GiB。vLLM 启动时会预留一块内存专门放 KV cache，这块内存能装多少 token，决定了单个请求最长多长、能同时处理多少请求。

可以对比一下 Qwen3-8B：权重 15.26 GiB，每个 token 的 KV cache 是 144 KiB。

## 4. 启动 vLLM

终端 1：

```bash
vllm serve /srv/spark/models/Qwen3-1.7B \
  --served-model-name qwen3-1.7b \
  --host 127.0.0.1 --port $PORT \
  --gpu-memory-utilization 0.08 \
  --max-model-len 8192
```

- 第一个参数是模型目录。写成模型 ID（如 `Qwen/Qwen3-1.7B`）会去 Hugging Face 下载，在 Spark 上会失败。
- `--served-model-name`：请求里 `model` 字段要填的名字。
- `--host 127.0.0.1`：只接受本机的请求。不写时监听所有网卡，实验室局域网里的其他机器也能访问。
- `--gpu-memory-utilization 0.08`：本服务最多使用总内存的 8%，约 9.7 GiB。7 个组同时运行合计约 68 GiB，还有余量。默认值 0.9 会占掉约 110 GiB，其他组就启动不了。
- `--max-model-len 8192`：一个请求的输入加输出最多 8192 个 token。

第一次启动约 3 分钟，要加载权重、编译模型、捕获 CUDA graph。编译结果缓存在 `~/.cache/vllm`，之后再启动约 1 分钟。看到 `Application startup complete.` 就可以调用了。

### 读懂启动日志

等待时，对照下面这些行（已去掉每行开头的时间和进程信息）：

```text
Model loading took 3.22 GiB memory and 46.626242 seconds
torch.compile took 14.21 s in total
Graph capturing finished in 37 secs, took 0.43 GiB
Available KV cache memory: 2.85 GiB
GPU KV cache size: 26,704 tokens, Maximum concurrency for 8,192 tokens per request: 3.26x
Free memory on device (116.59/121.69 GiB) on startup. Desired GPU memory utilization is (0.08, 9.74 GiB). Actual usage is 5.87 GiB for consumed memory (weights + non-torch), 1.02 GiB for peak activation, ...
init engine (profile, create kv cache, warmup model) took 125.45 s (compilation: 14.21 s)
Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`. ...
Starting vLLM server on http://127.0.0.1:8103
```

- 权重占 3.22 GiB，和 `model_info.py` 估算的约 3.20 GiB 一致。
- `torch.compile` 和 CUDA graph：把模型编译成更快的 kernel，并把生成阶段的一串 GPU 调用录下来整体重放，减少 Python 和 kernel 启动的开销。只有第一次慢，之后读缓存。
- 9.74 GiB 预算的去向：权重和 CUDA 运行时等固定开销共 5.87 GiB，计算时的激活峰值 1.02 GiB，剩下的 2.85 GiB 全部给 KV cache。
- 2.85 GiB ÷ 112 KiB ≈ 26,700 个 token，和第 3 节的公式对得上（`python model_info.py /srv/spark/models/Qwen3-1.7B --kv-gib 2.85` 可以直接算）。
- `3.26x`：26,704 ÷ 8,192 ≈ 3.26，即同时有 3 个 8192 token 的满长请求时 KV cache 就满了。实际请求一般短得多，能同时处理更多。
- 采样参数：vLLM 默认使用模型目录里 `generation_config.json` 推荐的 temperature 等值，请求里传的值优先。

## 5. 调用服务

在终端 2 先看服务上有哪些模型：

```bash
curl http://127.0.0.1:$PORT/v1/models
```

发一个对话请求：

```bash
curl http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-1.7b",
    "messages": [{"role": "user", "content": "用一句话介绍 DGX Spark。"}],
    "max_tokens": 200
  }'
```

Qwen3 默认先思考再回答，返回的 `content` 以 `<think>` 开头。实测这 200 个 token 全部用在了思考上，回答没写出来。在请求里加 `"chat_template_kwargs": {"enable_thinking": false}` 可以关闭思考。

Python 的写法和调用在线 API 一样，只是 `base_url` 换成本地地址：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8103/v1", api_key="EMPTY")  # 8103 换成本组端口
resp = client.chat.completions.create(
    model="qwen3-1.7b",
    messages=[{"role": "user", "content": "用三句话介绍一下 vLLM。"}],
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

`api_key` 随便填：启动服务时没有加 `--api-key`，就不检查。

`chat_demo.py` 用流式输出打印回答，并统计速度：

```bash
cd ~/week6
python chat_demo.py --port $PORT
python chat_demo.py --port $PORT --thinking --prompt "9.11 和 9.9 哪个大？"
```

```text
[回答]
vLLM 是一个高效的模型推理库，专为大规模模型的部署和推理优化设计……

模型 qwen3-1.7b | 输入 20 token | 输出 76 token
TTFT 0.036 s | 生成速度 50.9 token/s | 总耗时 1.51 s
```

- TTFT（Time To First Token）：从发出请求到收到第一个 token 的时间，主要是 prefill，即处理整段输入的时间。
- 生成速度：收到第一个 token 之后每秒生成多少个，对应 decode 阶段。它的倒数是 TPOT（Time Per Output Token），这里约 20 ms。
- 第一次请求会偏慢，多跑几次看稳定的数值。

发请求时看一眼终端 1：服务每 10 秒打印一行统计，包括正在处理（Running）和排队（Waiting）的请求数，以及 KV cache 占用率（GPU KV cache usage）。

## 6. 改参数做实验

每次只改一个参数：在终端 1 按 Ctrl+C 停掉服务，改好参数重新启动，再对比启动日志和 `chat_demo.py` 的结果。

| 参数 | 作用 | 默认值 |
| --- | --- | --- |
| `--gpu-memory-utilization` | 本服务可用的内存比例，包括权重、激活和 KV cache | 0.9 |
| `--max-model-len` | 单个请求最多多少 token（输入 + 输出） | 模型配置里的值，Qwen3 为 40960 |
| `--max-num-seqs` | 同时处理的请求数上限 | 这台机器上为 256 |
| `--enforce-eager` | 不编译、不捕获 CUDA graph | 关闭 |
| `--reasoning-parser qwen3` | 把思考内容放进单独的 `reasoning` 字段 | 关闭 |
| `--enable-auto-tool-choice --tool-call-parser hermes` | 支持请求里的 `tools`，Agent 调用工具需要 | 关闭 |
| `--api-key` | 请求必须带上这个密钥。本机所有用户都能访问你的端口，长时间运行时建议加上 | 不检查 |
| `--generation-config vllm` | 不使用模型自带的推荐采样参数 | 使用模型自带的 |

### 实验 1：不写 `--max-model-len`

去掉 `--max-model-len 8192`，其他不变，服务会拒绝启动：

```text
ValueError: To serve at least one request with the model's max seq len (40960), (4.38 GiB KV cache is needed, which is larger than the available KV cache memory (2.73 GiB). Based on the available memory, the estimated maximum model length is 25584. Try increasing `gpu_memory_utilization` ... or decreasing `max_model_len` when initializing the engine.
```

上下文默认是 40960，一个满长请求需要 4.38 GiB KV cache（第 3 节算过），而 0.08 的预算只剩 2.73 GiB，连一个请求都放不下。解决办法是调小 `--max-model-len`，或者调大 `--gpu-memory-utilization`。

### 实验 2：调大 `--gpu-memory-utilization`

把 0.08 改成 0.12：

| `--gpu-memory-utilization` | 内存预算 | KV cache | 可缓存 token | 8192 长度请求的最大并发 |
| --- | --- | --- | --- | --- |
| 0.08 | 9.74 GiB | 2.85 GiB | 26,704 | 3.26x |
| 0.12 | 14.6 GiB | 8.64 GiB | 80,864 | 9.87x |

权重和固定开销不变，多给的内存基本都变成了 KV cache。看完改回 0.08，给其他组留出内存。

### 实验 3：`--enforce-eager`

| 模式 | 启动时间 | 生成速度 |
| --- | --- | --- |
| 默认（编译 + CUDA graph） | 第一次约 190 秒，之后约 70 秒 | 51 token/s |
| `--enforce-eager` | 约 50 秒 | 48 token/s |

eager 模式省掉了编译和 CUDA graph，启动快很多；在这台机器上，1.7B 单个请求只慢了约 5%。调试时可以先用 eager 模式，测性能时去掉。

### 实验 4：分开思考内容，打开工具调用

启动命令末尾加上：

```bash
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes
```

再运行 `python chat_demo.py --port $PORT --thinking`，思考内容会单独显示在 `[思考]` 下。在 Python 里，思考内容在 `message.reasoning`（流式输出时为 `delta.reasoning`），`message.content` 只剩回答。

测试工具调用：

```bash
curl http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-1.7b",
    "messages": [{"role": "user", "content": "北京现在几点？"}],
    "tools": [{"type": "function", "function": {
      "name": "get_time",
      "description": "查询某个城市的当前时间",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

返回的 `finish_reason` 为 `tool_calls`，`tool_calls` 里是 `get_time` 和参数 `{"city": "北京"}`。模型只给出要调用的工具和参数，vLLM 把它解析成 `tool_calls`；工具由你的程序执行，再把结果发回模型，流程和第 3 周的 Function Calling 相同。

### 实验 5（选做）：同时发多个请求

vLLM 自带压测工具。先一次只发 1 个请求，共发 8 个：

```bash
vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:$PORT --endpoint /v1/chat/completions \
  --model qwen3-1.7b --tokenizer /srv/spark/models/Qwen3-1.7B \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --num-prompts 8 --max-concurrency 1
```

再改成 `--num-prompts 16 --max-concurrency 4`，对比结果里的这几项：

| 同时请求数 | 总生成速度（Output token throughput） | 平均 TTFT | 平均 TPOT |
| --- | --- | --- | --- |
| 1 | 49.6 token/s | 43 ms | 20.0 ms |
| 4 | 222.8 token/s | 106 ms | 17.2 ms |

4 个请求一起处理时，总生成速度约是 1 个时的 4.5 倍，每个请求的 TPOT 并没有变长。decode 阶段的瓶颈是读权重，读一遍权重可以同时为多个请求各算一个 token，这就是 batching；代价是 TTFT 变长了。课上并发不要超过 4，以免影响其他组。

## 7. 结束实验

- 在终端 1 按 Ctrl+C 停止服务，再用 `free -h` 确认内存已经释放。
- 每组同时只运行一个服务：同组成员共用账号，也共用端口。
- 课上 `--gpu-memory-utilization` 保持 0.08。
- 课后不要让服务一直开着。需要长时间运行时，放在 tmux 里并加上 `--api-key`，用完关掉。

## 常见问题

**`Address already in use`**：端口被占用。可能是自己之前的服务没关：用 `ps -u $USER -f | grep "vllm serve"` 找到进程，`kill <PID>` 结束；也可能是端口填错了，和别组冲突。

**`Free memory on device ... is less than desired GPU memory utilization`**：剩余内存不够你申请的比例。用 `free -h` 看剩余内存，调小 `--gpu-memory-utilization`。注意不写这个参数时默认是 0.9。

**`To serve at least one request with the model's max seq len ...`**：KV cache 放不下一个满长请求，见实验 1。

**连不上服务**（curl 报 `Couldn't connect to server`，Python 报 `Connection error`）：服务还没启动完（等到 `Application startup complete.`），或者端口不对。

**`The model ... does not exist`**：请求里的 `model` 要和 `--served-model-name` 一致。

**回答只有 `<think>` 部分，或者被截断**：思考用完了 `max_tokens`。关闭思考，或调大 `max_tokens`。

**`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`**：请求带了 `tools`，但服务没有打开工具调用，见实验 4。

**`ninja: not found` 或 FlashInfer 编译报错**：没有激活虚拟环境。先执行 `source ~/venvs/vllm/bin/activate` 再启动。

**PyTorch 显示的剩余内存比 `free -h` 少很多**：在统一内存上，`torch.cuda.mem_get_info()` 把系统的文件缓存也算作已用，而这部分内存需要时可以回收。看剩余内存以 `free -h` 的 available 列为准；vLLM 启动时已经按可回收的内存计算。

[返回课程目录](../README.md)
