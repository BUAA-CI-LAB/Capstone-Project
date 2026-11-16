# 第十一周：Agent 评估与 Trace

本周衔接第十周的 Agent Planning，为 Agent v2 构建统一任务集、评分器和可核对的执行记录。

配套实验默认使用 DGX Spark 上的 Qwen3-8B（BF16），服务名为 `qwen3-8b-bf16`。

- [课程课件](Agent_Evaluation.pptx)
- [第十一周作业](assignment.md)
- [第十一周作业说明](code/README.md)
- [Agent v2 参考实现与 Spark 运行说明](code/README.md)

## 课程内容

- 评估任务、试验、评分器与评估套件，能力评估与回归评估。
- Run、Trace、Thread 与单步、完整任务、多轮评估。
- 成功标准、参考依据、代码评分、人工复核与状态变化核验。
- 用统一任务集比较 Agent 版本，结合成功率、调用次数与延迟分析 Planning。
- 从失败 trace 中定位问题，修订工具、提示词或评估规则。

## 参考资料

- [智能体评估准备清单](https://www.langchain.com/blog/agent-evaluation-readiness-checklist)
- [智能体可观测性如何支撑评估](https://www.langchain.com/blog/agent-observability-powers-agent-evaluation)
- [AI 智能体评估详解](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)

[返回课程目录](../README.md)
