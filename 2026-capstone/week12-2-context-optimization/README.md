# 第十二周第 2 次课：上下文与缓存优化

- [课程课件](context-and-cache-optimization.pptx)
- [小组作业：计算与缓存方向的优化对比](assignment.md)

本次课程延续第 1 次课的计算优化，从 Agent 的角度讨论上下文本身：怎样让重复的前缀不再重复计算，怎样把用不上的内容移出上下文，以及怎样设计工具的返回结果。结合前缀缓存、LLMLingua、StreamingLLM、SnapKV 与 SWE-agent 等工作，理解缓存复用、上下文压缩与工具结果摘要的做法和代价。

## 课程内容

1. **Agent 上下文的开销**：上下文的组成与逐轮增长，长度带来的计算、显存、时延、费用与回答质量问题，以及复用、压缩、摘要与卸载三条优化路径。
2. **缓存复用**：前缀缓存的命中条件、对缓存友好的提示词组织、多实例部署的会话路由、KV 卸载与商业 API 的提示词缓存，以及非前缀复用 CacheBlend。
3. **上下文压缩**：LLMLingua 系列的提示词压缩，StreamingLLM、H2O、SnapKV 等 KV 压缩，以及对话历史压缩与缓存的关系。
4. **工具结果摘要**：工具返回为什么容易膨胀、SWE-agent 的返回格式设计与消融结果、截断过滤与分页，以及可恢复的压缩。
5. **设计与验证**：上下文预算、服务端与 Agent 两侧的测量指标，以及与期末项目的衔接。

## 本周作业

以小组为单位，在计算与缓存两个方向各完成一组优化前后的对比：计算方向比较标准 Attention 与 FlashAttention，缓存方向集成 StreamingLLM、SnapKV 或 LLMLingua-2 中的一项。具体要求见 [assignment.md](assignment.md)，12 月 6 日（周日）23:59 前在 SPOC 提交。

## 资料拓展

- [vLLM：Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)：块哈希、命中条件与适用场景。
- [SGLang](https://github.com/sgl-project/sglang)：RadixAttention 与缓存感知路由。
- [LMCache](https://github.com/LMCache/LMCache)：KV 卸载、跨实例共享与 CacheBlend 实现。
- [kvpress](https://github.com/NVIDIA/kvpress)：在 Transformers 上实现的 StreamingLLM、SnapKV 等 KV 压缩方法。
- [LLMLingua](https://github.com/microsoft/LLMLingua)：LLMLingua、LongLLMLingua 与 LLMLingua-2 的实现。
- [Anthropic：Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：Agent 上下文的组织、压缩与按需加载。
- [Manus：Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)：围绕 KV Cache 设计提示词、可恢复的压缩与目标复述。
- [Lost in the Middle · TACL 2024](https://arxiv.org/abs/2307.03172)：Figure 1 展示答案位置对准确率的影响。
- [LLMLingua · EMNLP 2023](https://arxiv.org/abs/2310.05736) 与 [LLMLingua-2 · ACL 2024 Findings](https://arxiv.org/abs/2403.12968)：基于小模型的提示词压缩。
- [StreamingLLM · ICLR 2024](https://arxiv.org/abs/2309.17453)：Figure 1 对比 attention sink 与几种窗口方案。
- [H2O · NeurIPS 2023](https://arxiv.org/abs/2306.14048)：heavy hitter 的观察与 KV 淘汰策略。
- [SnapKV · NeurIPS 2024](https://arxiv.org/abs/2404.14469)：Figure 1 展示观察窗口投票与 pooling。
- [CacheBlend · EuroSys 2025](https://arxiv.org/abs/2405.16444)：Figure 1 对比四种复用方式，Figure 6 展示选择性重算。
- [SWE-agent · NeurIPS 2024](https://arxiv.org/abs/2405.15793)：第 3 节介绍工具接口设计，消融实验比较返回格式对解决率的影响。

[返回课程目录](../README.md)
