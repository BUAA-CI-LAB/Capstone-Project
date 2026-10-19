# 第七周：模型服务与性能优化

本周以 vLLM 为主要例子，围绕模型服务、性能观测与服务优化展开。

## 10 月 20 日：Model Serving 优化

2026 年 10 月 20 日，周二。

- [课程课件：Model Serving 优化](model-serving-optimization.pptx)

本次课程讨论多个请求如何共享一个模型推理服务，重点理解
batching、concurrency、调度队列与请求管理之间的关系，
以及它们对吞吐、响应延迟和显存使用的影响。

### 课程内容

1. **组批与每轮调度**：Prefill 与 decode 的工作差异、
   连续组批、每轮 token 预算，以及 chunked prefill 的取舍。
2. **并发与性能取舍**：并发请求的统计位置，吞吐与响应时间的关系，
   以及输入、输出长度对服务负载的影响。
3. **队列与请求管理**：等待与运行状态、调度顺序、KV Cache、
   抢占与重算、请求结束与取消，以及过载和重试。
4. **性能观测与调优思路**：TTFT、ITL 与尾部延迟，
   并通过并发排队、长输入干扰和 KV 压力三个例子说明可能的优化方向。

### 资料拓展

- [vLLM：Architecture](https://docs.vllm.ai/en/stable/design/arch_overview/)：
  API Server、EngineCore 与 Worker 的职责，以及请求执行链路。
- [vLLM：Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)：
  Chunked prefill、抢占重算及相关参数的性能取舍。
- [vLLM：Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)：
  Token 预算、序列数量限制和调度策略等配置项的定义。
- [vLLM：Metrics Design](https://docs.vllm.ai/en/stable/design/metrics/)：
  请求生命周期中的事件，以及 TTFT、ITL、排队和执行时间的统计边界。
- [vLLM bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/)：
  在线性能测试的请求发送方式、并发限制与结果指标。
- [Orca · OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)：
  按迭代调度请求，是理解连续组批的重要机制来源。
- [PagedAttention · SOSP 2023](https://doi.org/10.1145/3600006.3613165)：
  KV Cache 的分块管理，以及显存利用率与组批容量之间的关系。
- [Sarathi-Serve · OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal)：
  分块 prefill 与 decode 的协同调度，以及吞吐与延迟的取舍。

[返回课程目录](../README.md)
