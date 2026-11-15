# 第十周作业：量化模型部署与量化前后对比

## 任务

本周实验在 DGX Spark 上以小组为单位完成，**每组提交一份**。第 1～5 项必做；第 6、7 项是尝试项，鼓励各组尝试，是否完成不影响评分；第 8 项选做，完成有加分。

1. **准备环境并自测脚本。** 按 [README](README.md) 准备环境，运行 `python bench_quant.py --selftest`，输出全部为 PASS。
2. **部署两个模型。** 在 DGX Spark 上分别部署课程组提供的 Qwen3-8B（BF16，`/srv/spark/models/Qwen3-8B`）和 Qwen3-8B-FP8（`/srv/spark/models/Qwen3-8B-FP8`），一次只起一个服务。每次服务启动后完成两件事：
   - 记录启动日志中的三行：`Model loading took`、`Available KV cache memory`、`GPU KV cache size`；
   - 发送一条对话请求，确认返回的是通顺的句子。
3. **在两个工作点上测量。** 用 `bench_quant.py` 对两个模型各测两个工作点：并发 1，以及另一个大于 1 的并发数，优先选择本组项目预计使用的值（若项目仅使用并发 1，则选一个并发值用于观察组批效果，并写明理由）。共 4 次运行。脚本每次运行测出性能（TTFT、TPOT、输出吞吐）和精度（`gsm8k`、`arc_challenge` 两个任务的分数）。同一工作点的两次运行只允许 `--model` 与 `--weight-bits` 不同，其余参数必须相同。
4. **对比并给出结论。** 用 `--compare` 对比同一工作点下 BF16 与 FP8 的结果，在实验报告中给出：
   - 性能对比：TTFT、TPOT、输出吞吐的变化；
   - 内存对比：权重实际占用与 KV Cache 可用空间，数值取自第 2 项记录的启动日志；
   - 精度对比：两个任务的分数差，并与标准误差比较，判断差异是否超过噪声；
   - 速度上限（屋顶线）对照：实测解码速度达到「内存带宽 ÷ 权重体积」的百分之多少，差距来自哪里。
5. **写出选型结论。** 按课件 2.8 的五项（约束、预算、候选与淘汰、最终选择、复查条件），说明本组 Agent 准备使用哪个模型、哪种量化方案，并给出支撑结论的数据。
6. **（尝试）在本组任务集上统计 Agent 指标。** 第 3 项的精度评测不涉及工具调用，量化对本组 Agent 的影响需要在本组任务集上测量。可以尝试：用本组 Agent 在 BF16 与 FP8 两个服务上各跑一遍本组任务集，把调用记录保存为 trace 文件，运行 `bench_quant.py` 时用 `--agent-trace` 指定，脚本会统计工具调用成功率、结构化输出合法率、任务完成率和平均步数。若本组 Agent 使用 OpenAI 兼容的 `tools` 接口，沿用第六周的工具调用配置，在两个模型的启动命令中都加入 `--enable-auto-tool-choice --tool-call-parser hermes`，并保持其他服务参数一致。trace 文件每行一条 JSON，一行对应一个任务，至少包含以下字段：

   ```json
   {"task_id": "t01", "steps": ["…"], "tool_calls": [{"args_valid": true, "error": null}], "final_output": "{…}", "success": true}
   ```

   脚本的统计方式：一次工具调用的 `args_valid` 为真且 `error` 为空，记为成功；`final_output` 能解析为 JSON，记为结构化输出合法；`success` 为真，记为任务完成；`steps` 的长度记为步数。这一项不要求完成，未尝试的小组在实验报告中写「未尝试」即可；尝试后结果不完整或与预期不符的，如实记录。
7. **（尝试）给 FP8 模型开启 KV Cache 量化。** 第 2 项部署的 FP8 模型未启用 KV Cache 量化，KV Cache 默认仍按 16 比特存放。Agent 的上下文会随工具调用结果不断变长，KV Cache 的容量决定了能支持多长的上下文和多高的并发。可以尝试：
   - 用 FP8 模型再起一次服务（起服务前按 README「尝试」一节确认 `nvcc` 可用），在第 2 项的启动命令后加上 `--kv-cache-dtype fp8`，服务名改为 `qwen3-8b-fp8-kvfp8`；
   - 记录启动日志中的 `Available KV cache memory` 与 `GPU KV cache size`，与未开启时对比，说明同样的 KV Cache 空间能容纳的 token 数变为原来的多少倍；
   - 用与第 3 项相同的参数运行 `bench_quant.py --model qwen3-8b-fp8-kvfp8 --weight-bits 8`，至少完成并发 1 这个工作点；
   - 用 `--compare` 与未开启时的 FP8 结果对比，说明速度和精度各自有没有变化。

   这一项不要求完成，未尝试的小组在实验报告中写「未尝试」即可。服务无法启动或结果异常的，记录报错原文与 vLLM 版本。
8. **（选做，加分）自行量化一份 FP8，并与课程组提供的 FP8 对比。** 要求依次完成以下四步：
   - 按 [README](README.md) 的「选做」一节，在单独的环境里运行 `fp8.py`，得到 `Qwen3-8B-FP8-self`，记录量化耗时与产物体积；
   - 按课件 1.4 做离线加载自检，确认产物能加载、输出是通顺的句子（代码以 README「选做」一节为准，新版 transformers 需要 `return_dict=True`）；
   - 以 `qwen3-8b-fp8-self` 为服务名起服务，用与第 3 项相同的参数运行 `bench_quant.py`，至少完成并发 1 这个工作点；
   - 用 `--compare` 与课程组提供的 FP8 对比，在实验报告中列出两者的产物体积、权重占用、速度和精度，并分析差异的可能原因。

   结果没有达到预期时，如实记录现象和已排查的内容。

### 运行命令

先按 [README](README.md) 起好服务，`--served-model-name` 要与下面的 `--model` 一致；再在另一个终端运行评测脚本。

```bash
cd ~/week10
source ~/venvs/vllm/bin/activate
export PORT=8103  # 换成本组端口；数据集设置按 README 执行

# BF16 服务运行期间
python bench_quant.py --model qwen3-8b-bf16 --weight-bits 16
python bench_quant.py --model qwen3-8b-bf16 --weight-bits 16 --concurrency 8

# 停掉 BF16 服务，起 FP8 服务后，只改 --model 与 --weight-bits
python bench_quant.py --model qwen3-8b-fp8 --weight-bits 8
python bench_quant.py --model qwen3-8b-fp8 --weight-bits 8 --concurrency 8

# 对比同一工作点下的两次结果
python bench_quant.py --compare results/run_qwen3-8b-bf16_c1.json results/run_qwen3-8b-fp8_c1.json
python bench_quant.py --compare results/run_qwen3-8b-bf16_c8.json results/run_qwen3-8b-fp8_c8.json
```

命令中的 `--concurrency 8` 是示例，请按第 3 项选择第二个并发值；修改后，比较命令中的 `_c8` 文件名也要同步修改。测量终端先按 README 准备数据集。

| 参数 | 含义 |
| --- | --- |
| `--model` | 必填。服务的 `--served-model-name` |
| `--weight-bits` | 必填。BF16 填 16，FP8 填 8，脚本用它估算屋顶线 |
| `--concurrency` | 并发数，默认 1 |
| `--base-url` | 服务地址，默认 `http://127.0.0.1:$PORT`（`PORT` 为 8100 + 组号）；末尾不加 `/v1` |
| `--tokenizer` | 分词器目录，默认 `/srv/spark/models/Qwen3-8B`，各次运行使用同一份 |
| `--name` | 结果文件名前缀，默认为「模型名_c并发数」，一般不需要修改 |
| `--agent-trace` | 本组的 trace 文件，可用绝对路径；相对路径从 `~/week10` 解析。不指定则跳过 Agent 指标 |

负载参数使用默认值，各次运行保持不变：输入 1024、输出 128、32 个请求、温度 0、随机种子 1234；精度评测使用 `gsm8k` 与 `arc_challenge`，每个任务 100 个样本、5-shot。确需调整时，使用 `--input-len`、`--output-len`、`--num-prompts`、`--eval-tasks`、`--eval-limit` 等参数，所有运行取相同的值，并在实验报告中写明。全部参数见 `python bench_quant.py -h`。

在 `~/week10` 运行命令时，脚本把结果保存在 `~/week10/results/`。每次运行生成三类文件：`perf_<名称>.json`（`vllm bench serve` 的原始输出）、`eval_<名称>*.json`（`lm_eval` 的原始输出，具体后缀以实际文件为准）和 `run_<名称>.json`（汇总结果）。

本周作业的重点是用同一套参数测出量化前后的差别，并据此做出选择；FP8 的提速倍数不作为评分依据。评分主要看三点：对比是否同口径，结论是否有数据支撑，未跑通或出现退化的部分是否如实记录。某一步未跑通时，提交已有结果、报错与排查记录，不以是否得到预期提速作为评分标准。

## 提交内容

每组提交一份，包含：

- 一份实验报告，**Word、PDF 或 Markdown 格式均可，文件名自定**。报告内容可参考下方结构。
- `results/` 目录：脚本生成的原始结果，不要手工修改。完整完成必做实验时，应包含 4 次运行各自的 `run_*.json`、`perf_*.json` 和 `eval_*.json`，即 `qwen3-8b-bf16_c1`、`qwen3-8b-fp8_c1`、`qwen3-8b-bf16_c<N>`、`qwen3-8b-fp8_c<N>` 四组，`<N>` 为本组选择的并发数。
- 尝试了第 6 项的小组：附上 BF16 与 FP8 各自的 trace 文件。
- 尝试了第 7 项的小组：附上开启 KV Cache 量化后的结果文件。
- 完成第 8 项的小组：附上自行量化模型的结果文件，以及产物目录下的 `quantize_report.json`。

未跑通的运行无需补造结果，在报告中注明缺失文件和原因。

无需提交模型权重、虚拟环境和未修改的 `bench_quant.py`。提交内容中不要包含账号、密码和令牌。

## 实验报告内容参考

报告结构可按本组实验情况调整，建议包含：

1. **基本信息与环境**：组号、成员姓名与学号、项目场景、设备、驱动与软件版本、模型路径、服务端口和内存预占比例。
2. **实验设置**：两个模型的完整启动命令、四次测量命令、第二个工作点的并发数及取值理由。
3. **部署与测量结果**：首条对话请求的返回、启动日志中的权重占用与 KV Cache 数据，以及两个工作点下的 TTFT、TPOT、输出吞吐、精度分数和标准误差。注明指标单位、样本数、测量时间、资源占用情况与原始结果文件。
4. **对比与选型结论**：分析性能、内存和精度变化，给出屋顶线对照，并按约束、预算、候选与淘汰、最终选择、复查条件说明本组的选型。
5. **尝试项与选做项**：尝试了 Agent 指标或 KV Cache 量化的，记录方法、结果与问题；未尝试的注明“未尝试”。完成自行量化的，补充量化耗时、产物体积、离线自检和与课程组 FP8 的对比。
6. **问题与观察**：如实记录未跑通的步骤、报错和排查过程，以及与预期不符的现象。可用表格、图表或截图辅助说明。

## 提交前自查

| 检查项 | 通过标准 |
| --- | --- |
| 服务可用 | `/v1/models` 返回的模型名与 `--served-model-name` 一致；首条对话请求返回通顺的句子，不是乱码或空输出 |
| 先预热再测量 | 测量前已发送过请求。首次请求包含编译与内存分配，直接计入会使 TTFT 偏高 |
| 同口径 | 同一工作点的两次运行只有 `--model` 与 `--weight-bits` 不同；`--compare` 没有报「口径不一致」，并已自行核对服务启动参数 |
| 两个工作点 | 并发 1 与本组并发各有 BF16、FP8 两组结果，共 4 次运行 |
| 精度有基线 | BF16 的 `lm_eval` 分数也已测出；`--eval-limit` 在各次运行中取值相同 |
| 差异与噪声 | 每个任务 100 个样本时，1～2 个百分点的分数差通常在标准误差之内，此时结论写「没有测出差异」 |
| 内存数据 | 取自启动日志的 `Model loading took` 与 `Available KV cache memory`。DGX Spark 上 `nvidia-smi` 的显存一栏可能显示 `Not Supported` 或 `N/A`，而且 vLLM 启动时即按预占比例占满内存池，服务总占用在量化前后几乎相同，不能作为对比依据 |
| 尝试项（第 6、7 项） | 尝试了的小组已填写对应一节；未尝试的小组已注明「未尝试」 |
| 原始结果 | `results/` 下保留脚本生成的原始 JSON，没有手工修改 |
| 版本信息 | 实验报告中写明 vLLM、lm_eval、驱动版本与模型路径 |
| 收尾 | 使用本组账号和端口；实验结束后已停止本组服务，没有遗留占用内存的进程 |

截止时间与提交入口以课程通知为准。

[返回本次课程](README.md)
