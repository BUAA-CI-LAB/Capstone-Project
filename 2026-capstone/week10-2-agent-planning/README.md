# 第十周第 2 次课：Agent Planning

本周以 LangGraph 为主要例子，围绕任务分解、状态流转与恢复机制展开。

- [课程课件：Agent Planning](agent-planning.pptx)

本次课程以测试报告助手为贯穿案例，讨论如何读取报告、计算通过率、
检索失败原因，并在用户确认后发布报告。
重点理解显式计划如何指导多步执行，工具观察如何驱动重规划，
以及任务中断或失败后如何保留进度并继续执行。

## 课程内容

1. **任务分解与显式计划**介绍子任务的输入、依赖与完成条件，
   比较 ReAct 式循环与 Plan-and-Execute，说明 Planner、Executor、
   Replanner 的分工，以及结构化计划校验、计划粒度与执行预算。
2. **状态流转与应用控制**介绍 LangGraph 的 State、Node、Edge，
   通过计划执行过程说明节点的局部更新、reducer 的覆盖与追加、
   条件路由和结束判定，以及模型决策与应用控制的职责。
3. **失败处理与恢复机制**讨论有限重试、重规划、检查点续跑与人工介入，
   介绍 checkpointer、Store、SQLite 持久化和 `thread_id`，
   并结合节点重执行说明外部副作用、稳定操作 ID、幂等性与未知结果的处理。
4. **课堂练习与项目衔接**通过状态合并推演、故障判断与故障注入检查执行路径，
   在已有 Agent 上增加显式计划、有限预算与恢复路径，形成 Agent v2，
   并保留计划、工具观察、失败类型和结束原因，供第十一周评估与 Trace 分析。

## 资料拓展

- State、Node、Edge、reducer、条件边与图执行的基本概念。
  [LangGraph Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api)
- Planner、Executor 与 Replanner 的完整示例，以及剩余计划和执行历史的更新。
  课件采用固定提交中的历史版本，运行时需结合当前接口与依赖版本适配。
  [官方 Plan-and-Execute 历史教程](https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb)
- 检查点、super-step、`thread_id`、状态历史，以及内存和 SQLite 等持久化后端。
  [LangGraph Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- `RetryPolicy`、异常筛选、节点重执行，以及外部写入的幂等性和结果确认。
  [LangGraph Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)
- 使用 `interrupt()` 暂停执行，通过同一线程与 `Command(resume=...)` 恢复人工输入。
  [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- 用小型 Notebook 练习图构建、状态合并与人工介入，可重点阅读 Module 1—3。
  [LangChain Academy](https://github.com/langchain-ai/langchain-academy)
- 显式拆分任务、制定计划并按子任务求解的研究背景。
  [Plan-and-Solve Prompting · ACL 2023](https://arxiv.org/abs/2305.04091)
- 将规划推理与外部观察解耦的架构思路，以及相应的模型调用成本分析。
  [ReWOO · 2023](https://arxiv.org/abs/2305.18323)

[返回课程目录](../README.md)
