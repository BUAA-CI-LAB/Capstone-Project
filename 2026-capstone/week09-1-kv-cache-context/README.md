# 第九周第 1 次课：KV Cache 与 Context

2026 年 11 月 3 日，周二。

- [课程课件](kv-cache-and-context.pptx)
- [实验任务指导：Serving 配置对比](assignment.md)

本次课程延续第七周的模型服务与性能优化，讨论 KV Cache 怎样保存和复用历史计算，以及上下文长度怎样影响内存占用和推理开销。结合 GQA、PagedAttention 和前缀缓存，理解模型结构、缓存管理与服务配置之间的关系。

## 课程内容

1. **缓存怎样工作**：每层、每个 KV head 保存的状态，Prefill 与 Decode 中的缓存产生、读取和增长，以及连续生成与新请求之间的区别。
2. **状态占多少资源**：KV 容量的组成、GQA 的共享结构与取舍，按模型配置估算单 token 成本，以及长度、常驻序列数量、计算和带宽的影响。
3. **状态怎样存放**：有效数据与预留空间、分页与块映射、按需分配、前缀缓存，以及请求结束后的回收和空间不足时的处理。
4. **Context 怎样配置**：Agent 多轮请求的状态增长、输入与输出的长度预算，以及上下文长度、客户端并发和服务端组批的配置取舍。

## 课后实验

在 **DGX Spark** 上沿用已有的本地 vLLM 服务，比较不同 **batching、并发与 context 配置**，提交一份 **Serving 对比记录**。实验需要说明配置和负载条件，记录吞吐与延迟，并根据结果为本组 Agent 推荐配置。

可以继续使用[第七周的测量工具与指标说明](../week07-2-vllm-serving-metrics/README.md)，也可以使用 vLLM 官方测试工具。具体任务、提交内容与简短报告格式见 [assignment.md](assignment.md)。

## 资料拓展

- [Transformers：How caching works](https://huggingface.co/docs/transformers/cache_explanation)：K/V 的保存、追加与跨步复用。
- [GQA · EMNLP 2023](https://aclanthology.org/2023.emnlp-main.298/)：Figure 2 展示 KV head 共享结构，Table 1 比较效率与模型表现。
- [PagedAttention · SOSP 2023](https://doi.org/10.1145/3600006.3613165)：分块管理、共享前缀，以及空间利用率与组批容量之间的关系。
- [vLLM：Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)：跨请求复用相同前缀的 K/V，以及适用场景。
- [Sarathi-Serve · OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal)：Prefill 与 Decode 的执行特点，以及组批对两阶段的不同影响。
- [vLLM：Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)：上下文长度、序列数量和每轮 token 预算等参数定义。
- [vLLM bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/)：在线性能测试的负载设置、统计指标与结果保存。

[返回课程目录](../README.md)
