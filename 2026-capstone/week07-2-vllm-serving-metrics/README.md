# 第七周第 2 次课：vLLM Serving 指标与优化实测

本讲在 DGX Spark 上以 Qwen3-1.7B 为对象，统一 TTFT、TPOT、ITL、吞吐与 KV Cache 的测量口径，并通过单变量对比理解并发、前缀缓存、每轮 token 预算、KV 容量和编译模式对服务表现的影响。

## 文件

- [课程课件](vllm-serving-metrics-lab.pptx)：指标定义、测量方法、实验设计与调优决策。
- [bench_lab.py](bench_lab.py)：流式逐 token 计时工具，仅使用 Python 标准库；可统计 TTFT、TPOT、ITL、端到端时延和输出吞吐。
- [第七周作业](assignment.md)：将本组 Agent 框架接到本地 vLLM 服务，完成固定任务测试并收集服务 metrics。

相关背景见[第七周第 1 次课：模型服务与性能优化](../week07-serving-and-optimization/README.md)。登录 DGX Spark 时请使用课程分配的小组账号，连接方式见 [DGX Spark 登录手册](../dgx-spark-login.md)。

## 使用前准备

1. 使用本组账号登录 DGX Spark，不要使用其他小组的账号、目录或进程。
2. 使用课程指定的模型与 vLLM 环境。各组使用约定端口，启动前先确认端口未被占用，也不要终止其他小组的服务。
3. 每组实验只改变一个服务端参数；更换参数时重新启动服务，并记录完整启动命令。

## 启动服务

先设置本组端口，再启动基础配置：

```bash
export PORT=81XX   # 替换为本组约定端口

vllm serve /srv/spark/models/Qwen3-1.7B \
  --served-model-name qwen3-1.7b \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.08 \
  --max-model-len 8192
```

服务启动后，在另一个终端运行测量。下面示例测量并发为 4 的一组请求：

```bash
python bench_lab.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model qwen3-1.7b \
  --concurrency 4 \
  --num-prompts 16 \
  --max-tokens 128 \
  --label c4 \
  --output outputs/c4.json
```

主要参数：

- `--concurrency`：同时在途的请求数。
- `--num-prompts`：本组请求总数。
- `--max-tokens`：每个请求最多生成的 token 数。
- `--shared-prefix-words 400`：在每个请求前加入共享长前缀，用于 Prefix Caching 对比。
- `--output`：保存本组测量明细的 JSON 文件。

输出包括请求成功/失败数，以及 TTFT、TPOT、端到端时延的 mean、P50、P95、P99 和输出吞吐。请将个人或小组的原始测量输出保留在自己的作业目录中，不要提交到本讲目录。

## 实验任务

基础负载和参数以课件为准。每组配置先预热，再在相同请求集下测量。

| 实验 | 只改变的变量 | 固定条件 | 重点记录 |
| --- | --- | --- | --- |
| 一：并发扫描 | 并发 1 / 4 / 8 / 16 | 输入 512、输出 128 token | 输出吞吐、TTFT、TPOT、ITL P99、Running / Waiting |
| 二：Prefix Caching | 关闭 / 开启前缀缓存 | 共享前缀约 400 词、并发 1 | 首个与后续请求 TTFT、命中率、输出吞吐 |
| 三：每轮 token 预算 | `--max-num-batched-tokens` 1024 / 8192 | 并发 4、输入约 2048、输出 64 token | TTFT 与 ITL 的尾延迟、输出吞吐 |
| 四：KV 容量 | `--gpu-memory-utilization` 0.08 / 0.16 | 并发 8 与 16 | KV token 总量、TTFT P99、Waiting 与抢占情况 |
| 五：编译模式 | 默认模式 / `--enforce-eager` | 并发 1 与 4 | 启动耗时、单流速度、TPOT 与输出吞吐 |

## 记录与分析要求

每次测量至少记录：

- 模型、vLLM 版本、完整服务启动命令和客户端测量命令；
- 请求总数、成功数、失败数、并发数，以及输入/输出 token 数；
- TTFT、TPOT、ITL、端到端时延和输出吞吐，并同时给出 mean 与尾部分位数；
- 与实验变量直接相关的服务端资源指标。

分析时注意：

- 吞吐与延迟必须成对报告；并发提高总吞吐，不代表单个请求更快。
- 改变配置后要在相同负载下复测，不能用不同输入长度、输出长度或请求数的结果直接比较。
- `bench_lab.py` 与 `vllm bench serve` 的测量位置和统计口径不同，同一张表中不要混报。
- 分位数需要足够的样本量；若两组差异接近测量波动，应增加请求数复测后再下结论。

[返回课程目录](../README.md)
