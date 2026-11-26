# 第十二周第 1 次课：计算优化

- [课程课件](compute-optimization.pptx)

本次课程讨论 Attention 前向计算的优化，只涉及推理时执行的前向部分。从标准实现的访存瓶颈出发，介绍 FlashAttention、PagedAttention 与 RadixAttention 分别改进了哪一步，以及在推理框架中怎样确认和验证这些优化。

## 课程内容

1. **Attention 前向的瓶颈**：三步计算之间 N×N 中间矩阵的读写、GPU 存储层次，以及 Prefill 与 Decode 的不同瓶颈。
2. **FlashAttention**：分块与融合、online softmax、前向算法与 IO 复杂度，FlashAttention-2 / 3 的并行划分，以及作为 attention backend 接入推理框架的方式。
3. **PagedAttention**：按块表读取分散存放的 K/V、间接寻址的代价、长上下文 Decode 的 split-KV 并行，以及分页 K/V 与 FlashAttention 类 kernel 的结合。
4. **RadixAttention**：前缀复用的场景、Radix 树的插入与淘汰、命中后的前向计算、缓存感知调度、与 vLLM 自动前缀缓存的对比，以及选用与验证方法。

## 本周作业

以小组为单位，在计算与缓存两个方向各完成一组优化前后的对比，其中计算方向比较标准 Attention 与 FlashAttention。具体要求见[第 2 次课的作业说明](../week12-2-context-optimization/assignment.md)，12 月 6 日（周日）23:59 前在 SPOC 提交。

## 资料拓展

- [FlashAttention 仓库](https://github.com/Dao-AILab/flash-attention)：前向 kernel 实现，以及支持分页 K/V 的 `flash_attn_with_kvcache` 接口。
- [PyTorch：scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：自动选择 Flash、Memory-Efficient 等实现。
- [Flash-Decoding](https://pytorch.org/blog/flash-decoding/)：长上下文 Decode 的 split-KV 并行与合并。
- [vLLM：Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)：块哈希、命中条件与适用场景。
- [SGLang](https://github.com/sgl-project/sglang)：RadixAttention、attention backend 与服务启动参数。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：分页 K/V 与可变长度 Attention kernel。
- [FlashAttention · NeurIPS 2022](https://arxiv.org/abs/2205.14135)：Figure 1 展示分块与存储层次，Theorem 2 给出 IO 复杂度。
- [FlashAttention-2 · ICLR 2024](https://arxiv.org/abs/2307.08691)：Algorithm 1 为前向算法，Figure 2、3 展示并行与 warp 划分。
- [PagedAttention · SOSP 2023](https://doi.org/10.1145/3600006.3613165)：Figure 5 展示分块读取，消融实验给出 attention kernel 的开销。
- [SGLang · NeurIPS 2024](https://arxiv.org/abs/2312.07104)：RadixAttention 的树结构、LRU 淘汰与缓存感知调度。

[返回课程目录](../README.md)
