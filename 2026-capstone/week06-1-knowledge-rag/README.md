# 第六周第 1 次课：Knowledge 与 RAG

2026 年 10 月 13 日，周二。

- [课程课件](knowledge-and-rag.pptx)
- [参考代码与运行说明](code/README.md)
- [课后作业与提交说明](assignment.md)

本次课程讨论外部知识怎样进入 Agent，以及检索得到的材料怎样支持回答。延续第五周的 Memory 与工具，重点关注知识处理、检索、上下文组织和来源引用之间的架构关系。

## 课程内容

1. **知识进入系统**：文档解析、切分、索引，以及来源与版本记录。
2. **检索与证据组织**：词项与向量检索、混合召回、重排、跨粒度组织、上下文与引用。
3. **Agent 如何检索**：按需查询、补充证据、验证和停止条件。
4. **系统设计与评估**：组件分工、知识更新、检索与生成效果评估，以及失败定位。

## 参考实践：Agent v1

[code/](code/README.md) 提供一个简单的项目文档助手。它加载用户保存的回答偏好，通过 `read_file` 获取任务，再通过 `search_knowledge` 查阅资料，给出端口报错的处理建议。知识库包含六份简短的教学资料，也保留了一份更综合的可选任务。

参考实现使用 Qwen2.5-3B-Instruct 和 BGE-small-zh-v1.5，可以复用已有 Python 环境。准备模型、执行命令和查看调用记录的方法见代码目录的 README。模型可能遗漏信息或生成不准确的引用，实际依据需要结合工具返回的原文核对。

课后实践围绕 **Memory、至少两种 Tool 和一个小型知识库** 展开。**提供的代码只是一个非常简单的参考**，可以按自己的想法自由修改、拓展，也可以独立设计实现，不必依托这份代码。场景、资料、工具和拓展方向自选；提交方式见 [assignment.md](assignment.md)。

## 参考资料

- [RAG 原始论文](https://arxiv.org/abs/2005.11401)
- [Sentence Transformers：Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [RAPTOR](https://arxiv.org/abs/2401.18059)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [ALCE](https://aclanthology.org/2023.emnlp-main.398/)
- [RAGFlow](https://github.com/infiniflow/ragflow)

[返回课程目录](../README.md)
