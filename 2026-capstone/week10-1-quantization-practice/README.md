# 第十周第 1 次课：模型量化实践与模型选择

- [课程课件](quantization-practice-and-model-selection.pptx)
- [第十周作业](assignment.md)

承接第九周第 2 次课的量化理论，本次课介绍量化产物的部署与模型选择，课件中的性能示例来自 RTX 4090 实测。配套实验为各组在 DGX Spark 上完成 BF16 与 FP8 的同口径对比。必做部分不需要自行量化：课程组已将 Qwen3-8B（BF16）和 Qwen3-8B-FP8 放在共享目录，各组搭好环境后依次启动 BF16 和 FP8 服务，用同一个评测脚本分别测量两个工作点，再撰写实验报告（Word、PDF 或 Markdown 均可，文件名自定）。选做部分要求自行量化一份 FP8：在单独的环境里运行 `fp8.py`，完成离线自检后，用相同的参数测量，并与课程组提供的 FP8 对比。

## 文件

- [课程课件](quantization-practice-and-model-selection.pptx)：量化的实现、加载与部署、根据端侧环境选择模型、本周作业，共 24 页。
- [assignment.md](assignment.md)：作业任务、提交内容与实验报告内容参考。
- [bench_quant.py](bench_quant.py)：评测脚本，脚本本身使用 Python 标准库，运行时调用已安装的 vLLM 和 lm_eval。一条命令完成健康检查、性能压测和精度评测；提供 trace 文件时另行统计 Agent 指标。模型、比特数、并发数等都用命令行参数指定，负载有默认值，不需要补代码。
- [requirements.txt](requirements.txt)：评测依赖（`lm_eval`）。
- 选做：[fp8.py](fp8.py)：自己把 Qwen3-8B 量化成 FP8，并用同一套口径测量。

相关背景见[第九周第 2 次课：模型量化技术](../week09-2-model-quantization/README.md)。登录 DGX Spark 时请使用课程分配的小组账号，连接方式见 [DGX Spark 登录手册](../dgx-spark-login.md)。

## 课程内容

1. **量化、加载与部署**：先讲量化的实现（一份配方加几行代码，对应 `fp8.py`），再分五步部署：检查量化产物与 `quantization_config`、离线加载自检、用 vLLM 起服务、读启动日志里的三行、发请求验证；以及报错对应回到哪一步。
2. **根据端侧环境选择模型**：查清 DGX Spark 的内存容量与带宽、算内存判断装不装得下、用「带宽 ÷ 权重体积」估速度上限、把 Qwen3 系列的几个候选放在一起比较并按场景做出选择、实测量化前后的对比；Agent 要补测的过程指标；选型结论的五项记录。
3. **本周作业**：任务与流程、环境准备与起服务、运行评测脚本、提交内容与实验报告。

## 使用前准备

1. 使用本组账号登录 DGX Spark（见[登录手册](../dgx-spark-login.md)），不要使用其他小组的账号、目录或进程；长任务放进 `tmux`。开跑前先看一眼资源占用：

   ```bash
   free -h                 # 统一内存剩余量（GPU 和 CPU 共用）
   ps -eo user,pid,rss,etime,cmd --sort=-rss | head
   ```

   把本目录传到 Spark（在自己电脑上，进入课程仓库的 `2026-capstone` 目录）：

   ```bash
   ssh spark "mkdir -p ~/week10"
   cd week10-1-quantization-practice
   scp README.md assignment.md bench_quant.py fp8.py requirements.txt quantization-practice-and-model-selection.pptx spark:~/week10/
   ```

   上面三条命令在自己电脑执行；`spark` 应按登录手册配置为本组账号。显式复制文件可避免重复上传时生成 `~/week10/week10-1-quantization-practice/`。上传后，在 Spark 上执行 `ls ~/week10/bench_quant.py ~/week10/assignment.md` 确认文件直接位于 `~/week10`。下文除另有说明外，命令均在 Spark 的 Bash 终端执行。

   本组端口是 **8100 + 组号**（第 3 组为 8103），所有组共用一台机器，端口不能重复。准备两个终端：一个运行服务，一个运行测量。

2. **环境沿用第六周的做法**，同组共用一个账号，装一份即可。首次安装时执行：

   ```bash
   python3 -m venv ~/venvs/vllm && source ~/venvs/vllm/bin/activate
   pip install -U pip uv -i https://mirrors.bfsu.edu.cn/pypi/web/simple
   uv pip install vllm==0.31.0 openai "lm-eval[api]" -i https://mirrors.bfsu.edu.cn/pypi/web/simple
   ```

   已有第六周环境的，直接激活并补装评测依赖，无需重建环境：

   ```bash
   source ~/venvs/vllm/bin/activate
   uv pip install "lm-eval[api]" -i https://mirrors.bfsu.edu.cn/pypi/web/simple
   ```

   确认 `python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"`，第六周的输出是 `0.31.0 2.13.0+cu130 True`。

   之后每个终端都先执行：

   ```bash
   source ~/venvs/vllm/bin/activate
   export PORT=8103          # 换成 8100 + 组号
   export HF_HUB_OFFLINE=1   # Spark 连不上 Hugging Face，不让 vLLM 尝试联网
   ```

3. 两个模型都在共享目录里，直接用，不用下载：

   ```bash
   MODEL_BASE=/srv/spark/models/Qwen3-8B          # BF16，约 16 GB
   MODEL_FP8=/srv/spark/models/Qwen3-8B-FP8       # FP8，约 9 GB
   ```

   起服务前先看一眼产物：`ls -lh $MODEL_FP8` 里分词器文件齐全、两个权重分片与 `model.safetensors.index.json` 对得上，`grep -A 8 quantization_config $MODEL_FP8/config.json` 能搜到。课程组提供的是 Qwen 官方的 FP8 产物，配置里是 `"quant_method": "fp8"`、`"weight_block_size": [128, 128]`（权重按 128×128 分块量化，激活值动态量化）；选做里用 `fp8.py` 自己量化出的产物是 `compressed-tensors` 格式、权重按通道量化。两种格式 vLLM 都能自动识别。

## 实验步骤

**先验证逻辑**，不需要 GPU，全部 PASS 再往下：

```bash
source ~/venvs/vllm/bin/activate
cd ~/week10
python bench_quant.py --selftest
```

**终端 1：起服务**，先在此终端设置上文的 `MODEL_BASE` 和 `MODEL_FP8`。BF16 与 FP8 各一次，一次只开一个（换 FP8 时把路径和服务名换成 `$MODEL_FP8`、`qwen3-8b-fp8`）：

```bash
vllm serve "$MODEL_BASE" --served-model-name qwen3-8b-bf16 \
    --port "$PORT" --host 127.0.0.1 --max-model-len 8192 --gpu-memory-utilization 0.2
```

**终端 2：验证服务。** 等终端 1 显示服务已就绪后，发送一条请求，确认返回的是通顺的句子。换成 FP8 后，下方 JSON 中的 `model` 也要改为 `qwen3-8b-fp8`：

```bash
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-bf16","max_tokens":40,"messages":[{"role":"user","content":"用一句话介绍你自己。/no_think"}]}'
```

- `--gpu-memory-utilization` 必须调低。Spark 多组共用，默认 0.9 会占掉约 110 GB，让别组的程序失败。本次 8B 实验使用 0.2。沿用第六周的共享资源约定，课下错峰运行，启动前检查剩余内存；资源不足时等待其他组结束。
- 端口用本组的 `$PORT`（8100 + 组号），不要占用别组的端口，也不要终止其他小组的服务。
- `--served-model-name` 要和 `bench_quant.py` 的 `--model` 完全一致，否则健康检查会报「服务里没有模型」。
- 不要手动指定量化方法，vLLM 读 `config.json` 里的 `quantization_config` 自动识别。
- 服务日志里的 `Model loading took … GiB`、`Available KV cache memory`、`GPU KV cache size` 三行记下来，实验报告要用。

**精度评测前准备数据集。** 共享模型不包含各账号的评测数据缓存。测量终端首次获取数据时，可按 Spark 实测记录使用镜像：

```bash
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
export HF_ENDPOINT=https://hf-mirror.com
```

随后运行作业中的测量命令，由 lm_eval 获取 `gsm8k` 和 `arc_challenge`。若镜像不可用，先由课程组协助把这两个数据集的缓存放到本组账号的 `~/.cache/huggingface/datasets`，再设置 `HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1` 运行。不要在尚无缓存时直接开启离线评测。服务终端仍可保持离线模式。

**终端 2：完成四次测量。** 进入 `~/week10`，按 [assignment.md](assignment.md) 的运行命令操作：BF16 服务运行时测并发 1 和本组选择的第二个并发；在终端 1 停掉 BF16、启动 FP8 后，再测相同的两个工作点。最后分别比较两个工作点下的 BF16 与 FP8 结果。

同一工作点的两次运行保持负载、分词器、服务参数和评测设置一致，仅切换模型及对应的 `--weight-bits`。`--compare` 会检查结果中记录的负载与评测参数，服务启动参数仍需自行记录和核对。本周脚本的 `--base-url` 使用 `http://127.0.0.1:$PORT`，末尾不带 `/v1`；脚本会自行追加 `/v1/models`、`/v1/chat/completions` 和 `/v1/completions`。这与第七周 `bench_lab.py` 的地址写法不同。脚本默认使用 `$PORT`；两个终端都必须设置本组端口，否则会回退到 8100。分词器默认使用 `/srv/spark/models/Qwen3-8B`。所有相对路径（包括 `results/`、`--agent-trace`）均相对于执行命令时的目录；统一在 `~/week10` 运行，结果就保存在 `~/week10/results/`。

**记录测量条件。** 同一配置在其他组运行任务时可能明显变慢。尽量连续完成对比，记录测量时间与资源占用；差异接近波动时重复测量，不只保留最好的一次。重复运行请用 `--name` 指定不同名称，避免覆盖已有的汇总和性能结果。课件中的 4090 数据用于讲解，不作为 Spark 作业的速度目标。

**关于内存占用**：GB10 上 `nvidia-smi` 的显存一栏可能显示 `Not Supported` 或 `N/A`，脚本里的 `peak_memory_gb()` 会返回 `None`。而且 vLLM 起服务时就按预占比例把内存池一次占满，服务运行时的总占用两次几乎一样，不能当对比依据。实验报告里请用服务日志里的 `Model loading took … GiB`（权重实际占用）和 `Available KV cache memory`（KV 可用空间也会受运行开销和其他进程影响），并注明来源。

**收尾**：实验结束后停掉本组的服务（`ps -u $USER` 查看，`kill <PID>`），退出不用的 tmux 会话，不要遗留占用内存的进程。

## 选做：自己量化一份 FP8

选做要求依次完成四步：自行量化、离线自检、用相同参数测量、与课程组提供的 FP8 对比，具体要求见 [assignment.md](assignment.md) 第 8 项。下面是操作方法。

llmcompressor 要装在**单独的环境**里：它和 vLLM 对 `torch`、`transformers` 的版本要求不同，装进同一个环境可能把 torch 升级，导致 vLLM 报错、`torch.cuda.is_available()` 变成 `False`。

自行量化前先停止本组的 vLLM 服务，避免同时加载两份模型。

```bash
python3 -m venv ~/venvs/quant && source ~/venvs/quant/bin/activate
pip install -U pip uv -i https://mirrors.bfsu.edu.cn/pypi/web/simple    # 新环境里没有 uv，先装
uv pip install llmcompressor -i https://mirrors.bfsu.edu.cn/pypi/web/simple
cd ~/week10
python fp8.py --src /srv/spark/models/Qwen3-8B --dst ~/models/Qwen3-8B-FP8-self
```

该配方不需要校准数据，耗时取决于设备和当时的资源占用。脚本结束时会打印耗时、产物体积（约 9 GB）以及 `config.json` 里是否有 `quantization_config`，并在产物目录写出 `quantize_report.json`。

量化完成后先做离线自检（课件 1.4）：仍在 `~/venvs/quant` 环境里，用 Transformers 直接加载产物并生成一句话，确认输出是通顺的句子。这个环境里的 transformers 较新，`apply_chat_template` 返回的是字典，按下面的写法取输入：

```python
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_PATH = str(Path.home() / "models" / "Qwen3-8B-FP8-self")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="auto")
msgs = [{"role": "user", "content": "用一句话介绍你自己。"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                 return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

自检完成后退出 Python 进程，释放模型占用，再运行 `deactivate` 退出量化环境。

先停止本组原有服务。在终端 1 执行 `source ~/venvs/vllm/bin/activate`，然后启动自行量化的模型：

```bash
vllm serve ~/models/Qwen3-8B-FP8-self --served-model-name qwen3-8b-fp8-self \
    --port "$PORT" --host 127.0.0.1 --max-model-len 8192 --gpu-memory-utilization 0.2

```

终端 2 保持测量环境与数据集设置，进入 `~/week10` 后运行：

```bash
cd ~/week10
python bench_quant.py --model qwen3-8b-fp8-self --weight-bits 8
python bench_quant.py --compare results/run_qwen3-8b-fp8_c1.json results/run_qwen3-8b-fp8-self_c1.json
```

两个 FP8 的量化方案不同，不预设速度或精度相同。记录差异，分析可能的原因（量化方案、分块粒度、vLLM 内核选择等）。

## 尝试：给 FP8 模型开启 KV Cache 量化

对应 [assignment.md](assignment.md) 第 7 项，鼓励尝试，不影响评分。这里的 FP8 模型并未启用 KV Cache 量化，KV Cache 默认仍按 16 比特存放；加上 `--kv-cache-dtype fp8` 后，KV Cache 也按 8 比特存放。

开启后 vLLM 要用 FlashInfer 即时编译 FP8 KV Cache 的内核，需要 CUDA 编译器 `nvcc`。起服务前先在终端 1 运行 `which nvcc`；找不到时执行下面两行，否则服务可能报 `FlashInfer backend is not available` 后退出：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

先停止本组原有服务，在终端 1 激活 `~/venvs/vllm` 并确认本组的 `PORT`，然后启动：

```bash
MODEL_FP8=/srv/spark/models/Qwen3-8B-FP8
vllm serve "$MODEL_FP8" --served-model-name qwen3-8b-fp8-kvfp8 \
    --port "$PORT" --host 127.0.0.1 --max-model-len 8192 --gpu-memory-utilization 0.2 \
    --kv-cache-dtype fp8

```

服务就绪后，在终端 2 运行（已激活 vLLM 环境并设置本组 `PORT`）：

```bash
cd ~/week10
python bench_quant.py --model qwen3-8b-fp8-kvfp8 --weight-bits 8
python bench_quant.py --compare results/run_qwen3-8b-fp8_c1.json results/run_qwen3-8b-fp8-kvfp8_c1.json
```

先看启动日志：`Available KV cache memory` 应与未开启时基本相同，`GPU KV cache size` 应明显变大。再看评测结果：速度和精度有没有变化。这个参数在不同 vLLM 版本和硬件上的支持情况不同，服务起不来时记录报错原文，不必反复尝试。

## 拓展（选做）

- **并发扫描**：并发从 1 到 32，看吞吐与单请求速度如何变化，说明瓶颈在带宽还是算力。
- **输入长度**：逐步增加输入长度，观察首字延迟。输入与输出 token 总数不能超过服务的 `--max-model-len`；默认上限 8192 时，不能直接使用 8192 token 输入再生成输出。
- **更长的上下文**：在开启 KV Cache 量化的服务上，把 `--max-model-len` 调大，用更长的输入测首字延迟与可并发数。
- **屋顶线**：实测解码速度占「273 GB/s ÷ 权重体积」的百分之几，差距来自哪里。

[返回课程目录](../README.md)
