# Agent v2 参考实现与评估

这个示例延续第六周的项目文档助手，读取任务文件、查询六份教学资料，并按要求保存说明。它增加了模型生成的显式计划、执行进度、工具出错后的重规划、有限预算，以及逐任务评分与 trace。

可以继续自己的项目，参考这里的接口组织评估。作业要求见[第十一周作业](../assignment.md)。

## 实现与文件

- `agent_v2.py` 提供 `baseline` 和 `planning` 两种运行模式。`baseline` 直接选择下一步动作，`planning` 先调用 Planner，执行器读取当前计划并报告已完成步骤，工具出错后调用 Replanner 修订剩余计划。
- `eval_agent.py` 按固定任务集调用 Agent，建立每次试验独立的工作目录、Memory 与故障状态，检查结果并保存原始记录。
- `tasks.jsonl` 是固定的 8 条示例任务，预期结果只供执行后的评分器读取。
- 第六周的 `agent_v1.py`、`knowledge/`、`workspace/` 提供原有资料与 `read_file`、Memory 读写、文档分块函数，通过模块导入复用。

本示例使用 Python 3.10 及以上版本的标准库调用 vLLM。服务端沿用课程环境，无需为客户端另外安装 Python 包。检索使用中文双字词项与英文词项的重合数排序，返回前三个片段，便于在 Spark 上复现实验。第六周示例使用的 BGE 向量检索可以继续用于本组项目。两种对比模式必须使用相同的检索实现与资料。

工具调用采用应用定义的 JSON 动作，由 Python 校验并执行。请求使用 `response_format={"type":"json_object"}`，关闭 Qwen3 thinking。完整请求、服务原始响应、原始文本和应用解析结果都会写入 trace。服务命令使用常规 Chat Completions 配置。[vLLM 结构化输出说明](https://docs.vllm.ai/en/stable/features/structured_outputs/)、[Qwen 部署说明](https://qwen.readthedocs.io/en/stable/deployment/vllm.html)

## 上传文件

以下命令在自己电脑的课程仓库 `2026-capstone` 目录执行。`spark` 按[登录手册](../../dgx-spark-login.md)配置为本组账号。

```bash
ssh spark "mkdir -p ~/week11/code/v1"
scp week11-1-agent-eval-trace/code/agent_v2.py week11-1-agent-eval-trace/code/eval_agent.py week11-1-agent-eval-trace/code/tasks.jsonl spark:~/week11/code/
scp week06-1-knowledge-rag/code/agent_v1.py spark:~/week11/code/v1/
scp -r week06-1-knowledge-rag/code/knowledge week06-1-knowledge-rag/code/workspace spark:~/week11/code/v1/
```

完整仓库已经在 Spark 上时，也可以进入仓库内本 `code/` 目录直接运行。程序先找当前目录的 `v1/`，再找仓库内的第六周代码。其他布局可用 `--v1-dir /实际路径/code` 指定。

## 启动服务

下文命令在 Spark 的 Bash 终端执行。沿用第六周或第十周已安装的 vLLM 环境，环境准备见[第十周运行说明](../../week10-1-quantization-practice/README.md)。开始前查看 `free -h`、`nvidia-smi` 与本组进程，使用本组端口与共享内存预算。

终端 1 运行课程组提供的 Qwen3-8B（BF16），模型目录为 `/srv/spark/models/Qwen3-8B`。

```bash
source ~/venvs/vllm/bin/activate
export PORT=8101  # 换成 8100 + 组号
export HF_HUB_OFFLINE=1

vllm serve /srv/spark/models/Qwen3-8B \
  --served-model-name qwen3-8b-bf16 --dtype bfloat16 \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 8192 --gpu-memory-utilization 0.2
```

等待 `Application startup complete.` 后，再在终端 2 检查模型并运行评估。启动服务前激活环境可让编译工具找到 `ninja`。服务名须与客户端的 `--model` 一致。

```bash
cd ~/week11/code
source ~/venvs/vllm/bin/activate
export PORT=8101  # 与终端 1 相同
curl -f "http://127.0.0.1:${PORT}/v1/models"

python eval_agent.py --selftest
python eval_agent.py --mode both --repeats 2 \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model qwen3-8b-bf16 --output results/agent-v2
```

脚本会先检查任务集、资料与服务，发一条预热请求，再开始计时。每条任务、每个版本各运行两遍，共 32 条记录。第二遍调换两版本的运行先后顺序。两版本同一遍使用相同 seed，第二遍 seed 加 1。

客户端未能完成任务时仍会保存 trace、状态与失败评分，随后继续其他任务。评分器自身抛错时记录 `evaluation_status=error`，并在汇总中列出。服务未就绪等运行前检查失败会直接报错，确认服务后重新运行即可。

### 常用参数

| 参数 | 默认值与含义 |
| --- | --- |
| `--mode` | `both`，也可单独运行 `baseline` 或 `planning` |
| `--tasks` | 脚本旁的 `tasks.jsonl` |
| `--repeats` | 2，每条任务、每个版本的试验次数 |
| `--output` | 未指定时使用脚本旁 `results/` 内的 UTC 时间戳目录。显式指定时按当前工作目录解析，已有目录会报错 |
| `--base-url` | `http://127.0.0.1:$PORT/v1`，未设置 PORT 时使用 8103，末尾包含 `/v1` |
| `--model` | `qwen3-8b-bf16` |
| `--temperature`、`--seed` | 0、1234 |
| `--max-tokens` | 每次模型请求最多生成 768 token |
| `--max-calls`、`--max-tools` | 每任务最多 16 次模型请求、10 次工具执行，两种模式相同 |
| `--max-replans` | 2，Planning 模式最多重规划两次，计入模型调用预算 |
| `--timeout`、`--request-timeout` | 每任务总时限 300 秒、单次请求时限 120 秒 |
| `--v1-dir` | 第六周 `agent_v1.py`、`knowledge/` 与 `workspace/` 所在目录 |

需要认证的服务可设置 `OPENAI_API_KEY` 环境变量。trace 保存请求体，认证头不写入记录。更换其他模型或服务时，需要核对其结构化输出和 `chat_template_kwargs` 支持情况。

## 示例任务与评分

| 编号 | 类别 | 核验内容 |
| --- | --- | --- |
| t01 | 单次检索 | 日志路径与来源 |
| t02 | 多步任务 | 读取报错、查询配置、给出端口修改、重启和日志路径 |
| t03 | 多步任务 | 读取综合求助，组合安装、配置、日志资料，识别未规定的保留天数 |
| t04 | 单次检索 | 筛选后数据的导出路径、编码与覆盖行为 |
| t05 | 资料不足 | 明确说明保留天数未规定，引用真实资料 |
| t06 | 文件缺失 | 实际读取 missing.txt，说明文件不存在 |
| t07 | 工具恢复 | 第一次检索注入 TimeoutError，后续成功检索并回答 |
| t08 | 写入产物 | 实际生成 notes/logging.md，内容包含日志路径与来源 |

`expected` 中的正则匹配关键命令、路径和必要信息，引用检查确认 id 出现在实际检索结果中，并满足该任务要求的来源。t08 还读取保存文件的真实内容，其他任务检查 `notes/` 中是否出现额外产物。必要工具只核对任务要求的行为，各工具的完整调用顺序由 Agent 决定。

评分器采用关键词和正则，中文同义表达可能需要人工复核。来源 id 的检查可以验证引用是否检索到，正文与来源是否语义一致仍需审查。至少查看两条逐任务得分，结合实际回答、资料片段和文件判断评分是否合理。

修正 `expected` 中的评分规则后，可以对整批原始记录重新评分。该命令会校验任务输入、类别和故障设置，并在新目录保存两种版本的全部评分、规则副本与哈希。输入或故障设置变化时，需要重新运行 Agent。

```bash
python eval_agent.py --regrade results/agent-v2 --tasks tasks.jsonl \
  --output results/agent-v2-reviewed
```

增加自选任务时，沿用以下结构，按项目需要扩展评分器。

```json
{"task_id":"t09","category":"single_lookup","input":"用户任务","expected":{"answer_all":["必要事实的正则"],"answer_none":["禁止出现的无依据结论"],"citation_groups":[["可接受的来源A","可接受的来源B"]],"required_tools":["search_knowledge"]}}
```

每个 `citation_groups` 子列表要求至少命中一个来源。`required_tools` 检查调用尝试，错误与成功另行记录。每条任务都从第六周 `workspace/` 的全新副本和固定偏好 Memory 开始。检索故障只在 t07 的工具实例中注入一次，随后自动清除。

## 查看结果

```text
results/agent-v2/
  manifest.json          # 环境、参数、代码、任务集与输入资料哈希
  tasks.jsonl            # 本次使用的固定任务集副本
  warmup.json            # 预热请求与响应，单独保存
  summary.json           # 按版本、类别汇总，finished 标记整批是否结束
  results.jsonl          # 每行一次试验，含检查项、失败原因与 trace 路径
  traces/<版本>/         # t01_r1.json 等完整执行记录
  workspaces/<版本>/     # 每次试验的输入、初始 Memory 与真实产物
```

trace 的 `events` 按发生顺序记录 Planner、Executor、Replanner 的请求与响应，以及工具参数、Observation、错误和耗时。LLM 事件中的 `state` 是该次调用前的状态，`plan_revision` 标识计划版本。`completed_steps` 是执行器声明的进度，结束时尚未声明完成的步骤保存在 `uncompleted_steps`，供人工结合 Observation 核验。任务评分检查最终答案与真实状态。工具异常在下一次规划中可见，重规划会保留历史观察并替换剩余计划。

运行状态包含 `completed`、`error`、`timeout`、`call_budget`、`tool_budget` 和 `replan_budget`。`completed` 说明模型通过输出校验并结束，任务是否成功由 `results.jsonl` 的 `score.success` 给出。

汇总中的任务成功率以已完成评分的试验为分母，另列试验数、评分覆盖数和评分错误数。格式合法率以全部试验为分母。工具成功率按参数有效且无执行错误的调用计数。效率均值包含失败和预算耗尽的试验，Token 用量来自服务返回的 usage，缺失时显示 `null`。脚本正常完成评估后，即使有任务未通过，也可以返回退出码 0。评分器报错则返回 1。

单条任务调试可以直接运行 Agent。先准备自己的工作目录，Memory 从其中的 `preferences.json` 读取，文件缺失时使用空列表。

```bash
cp -r v1/workspace debug-workspace
python agent_v2.py --task "查询运行日志路径并给出来源" \
  --workspace debug-workspace --mode planning \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --output debug-trace.json
```

实验结束后，在服务终端按 `Ctrl+C` 停止本组服务，并保存结果目录。
