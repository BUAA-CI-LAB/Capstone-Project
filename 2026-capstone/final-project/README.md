# 期末大作业说明

期末大作业要求各组在 DGX Spark 上本地部署推理服务与 Harness，提供 Coding Agent 服务。Harness 通过 OpenAI 兼容 API 请求本地推理服务，在命令行中完成编码任务，并用两层 Trace 同时记录 Agent 层调用链与服务层的 TTFT、TPOT 等指标。

- [期末大作业说明课件](final-project-guide.pptx)

## 主要内容

- 课设目标与系统架构：本地推理服务、Harness 与统一 Trace。
- 任务清单：Serving、Harness、Trace、主任务与统一评测为必做，模型量化、多模态与云端顾问协同为选做。
- 评估指标与提交材料：服务层与 Agent 层指标，以及 Baseline 与优化后的对比。
- 主任务（四选一）：网页游戏、网页桌面系统、3D 滚动网站、MIPS 流水线 CPU，每项给出开发环境、模块验收与开发流程。
- 统一评测：OdysseyBench 与 AppWorld 共 10 道固定题目，同一次运行报告模型能力与服务能力。

[返回课程目录](../README.md)
