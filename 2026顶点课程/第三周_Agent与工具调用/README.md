# 第三周：Agent 与 Tool Calling

- [Minimal Agent 与 Tool Calling 课件](Minimal%20Agent与Tool%20Calling.pptx)
- [完整实现](agent_v0.py)
- [课后作业](作业.md)

本周从第二周的 LLM 应用出发，补上"循环、状态、工具"，做出一个能跑通 **模型 → 工具 → Observation → 最终回答** 闭环的最小 Agent v0。协议采用 **Prompt 约定的 JSON 决策**，只依赖 Pydantic，用 CPU 或 GPU 均可；先用脚本化模型验证逻辑，再接入真实模型。

## 准备

复用前两周的 Python 环境和模型（Qwen2.5-1.5B-Instruct），在本目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

尚未准备环境或模型的同学，参照[第一周说明](https://github.com/BUAA-CI-LAB/Capstone-Project/blob/master/2026%E9%A1%B6%E7%82%B9%E8%AF%BE%E7%A8%8B/%E7%AC%AC%E4%B8%80%E5%91%A8_LLM%E6%8E%A8%E7%90%86/README.md)完成 PyTorch 安装（`python -m venv .venv` 建独立环境，不需要 conda），再下载模型到本目录：

```bash
python -m pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct')"
```

想对比不同模型规模（作业里的可选调参方向之一），可以再下载 3B/7B（`agent_v0.py` 的 hf 后端在 CPU 上用 bfloat16，7B 大约占 14GB 内存，下载前确认硬盘和内存足够）：

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='models/Qwen2.5-3B-Instruct')"
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='models/Qwen2.5-7B-Instruct')"
```

## 先验证逻辑（无需 GPU、无需下载模型）

```bash
python agent_v0.py --selftest
```

用脚本化模型运行固定任务集（T1–T8）和非法表达式检查，全部显示 PASS 表示 Agent Loop（状态、解析、执行与 Observation、停止条件）和工具层（`execute()`、`calculator` 的 ast 白名单求值）都按预期工作。

## 接入真实模型

以下命令中的 `--model` 请替换为模型的实际本地目录。首次运行会自动创建 `workspace/test_report.txt` 作为演示文件。使用 GPU 时将 `--device cpu` 改为 `--device cuda`。

```bash
python agent_v0.py --backend hf --model ./models/Qwen2.5-1.5B-Instruct --device cpu \
    --task "读取 test_report.txt，计算测试通过率" --output results/run.json
```

已用 vLLM 等 OpenAI 兼容服务时：

```bash
python agent_v0.py --backend openai --base-url http://localhost:8000/v1 --model <服务端模型名> \
    --task "计算 1024*1024" --output results/run_api.json
```

hf 后端默认**贪心解码**（同样输入、同样输出，方便复现和对比）；加 `--sample` 开启采样（可配 `--temperature`/`--top-p`/`--top-k`），适合多跑几次、观察结果的波动：

```bash
python agent_v0.py --backend hf --model ./models/Qwen2.5-1.5B-Instruct --device cpu --sample \
    --task "读取 test_report.txt，计算测试通过率" --output results/run_sample.json
```

常用参数：`--max-steps`（步数上限，默认 6）、`--max-failures`（失败预算，默认 2）、`--max-new-tokens`（默认 256）、`--workdir`（`read_file` 的工作目录）。不同任务使用不同文件名，避免覆盖。

## 查看结果

调用链保存在 `--output` 指定的 JSON 中。重点查看：`status`（`ok`、`max_steps`、`too_many_failures`、`no_progress`）、`answer`、`steps`，以及 `trace` 中每一步的 `decision`（工具名、参数或 final）、`observation`（`status` 与 `result` / `type` / `message`）和 `raw`（模型原文）。

小模型输出格式不稳定、甚至会编造工具调用结果，属于常见现象：用调用链定位卡在哪一步，再判断是模型能力、Prompt 还是解码策略（贪心 vs 采样）的问题。`read_file` 只能读取工作目录内的文件，`calculator` 只放行数字、四则运算与负号，不使用 `eval`。具体任务和提交格式见[作业.md](作业.md)。
