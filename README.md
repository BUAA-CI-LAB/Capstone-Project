# 顶点课程（Capstone Project）

顶点课程是北京航空航天大学计算机学院面向计算机拔尖计划大四学生开设的实践课程。学生围绕具体应用，在目标设备上部署大模型，并通过实验评估系统的表现。

2024 年首次开课时，学生在智能手机上部署大模型并开发应用。2025 年延续手机端实践，进一步探索如何在有限的算力和内存下优化推理。2026 年是开课第三年，学生在一体机上设计和部署智能体系统，让模型通过工具调用、知识检索和任务规划完成多步任务。

## 2026 年 · 在一体机上设计与部署智能体

下表按已发布的周次排列。各讲的课件、示例或实验步骤见对应课程页面，首次上机可先按登录手册连接一体机。[2026 年课程目录](2026-capstone/README.md) · [DGX Spark 登录手册](2026-capstone/dgx-spark-login.md)

| 周次 | 课程资料 | 学习与实践 |
| --- | --- | --- |
| 第 1 周 | [LLM 推理](2026-capstone/week01-llm-inference/README.md) · [作业](2026-capstone/week01-llm-inference/assignment.md) | 运行 Qwen 推理脚本，观察预填充和逐个生成 token 时隐藏状态与 KV Cache 的变化，记录推理耗时。 |
| 第 2 周 | [LLM 应用开发](2026-capstone/week02-llm-application/README.md) · [作业](2026-capstone/week02-llm-application/assignment.md) | 构建支持多轮对话的代码助手，处理流式响应并校验模型输出的 JSON。 |
| 第 3 周 | [Agent 与工具调用](2026-capstone/week03-agent-tool-calling/README.md) · [作业](2026-capstone/week03-agent-tool-calling/assignment.md) | 运行最小智能体 Agent v0，跟踪模型如何选择工具、读取执行结果并决定下一步。 |
| 第 5 周 | [智能体记忆、工具与 MCP](2026-capstone/week05-agent-memory-tools/README.md) | 了解智能体如何跨会话保存和检索记忆，以及如何通过 MCP 接入外部工具。 |
| 第 6 周第 1 讲 | [知识库与检索增强生成（RAG）](2026-capstone/week06-1-knowledge-rag/README.md) · [作业](2026-capstone/week06-1-knowledge-rag/assignment.md) | 以项目文档助手 Agent v1 为例，从知识库检索证据，用检索结果回答问题并标注来源。 |
| 第 6 周第 2 讲 | [DGX Spark 本地部署与 vLLM](2026-capstone/week06-2-dgx-local-serving/README.md) | 在一体机上启动 Qwen 模型服务，用 Python 发起请求，观察上下文长度与显存设置对运行的影响。 |
| 第 7 周第 1 讲 | [模型服务与性能优化](2026-capstone/week07-serving-and-optimization/README.md) | 以 vLLM 为例，分析多个请求如何排队和组批，以及调度方式对吞吐和延迟的影响。 |
| 第 7 周第 2 讲 | [vLLM 服务指标与优化实测](2026-capstone/week07-2-vllm-serving-metrics/README.md) · [作业](2026-capstone/week07-2-vllm-serving-metrics/assignment.md) | 在 DGX Spark 上测量首个 token 的等待时间、后续 token 的生成延迟和吞吐量，比较并发与前缀缓存等配置。 |
| 第 9 周第 1 讲 | [KV Cache 与上下文](2026-capstone/week09-1-kv-cache-context/README.md) · [实验指导](2026-capstone/week09-1-kv-cache-context/assignment.md) | 估算 KV Cache 的内存开销，比较不同上下文长度、并发数和组批配置下的服务表现。 |
| 第 9 周第 2 讲 | [模型量化技术](2026-capstone/week09-2-model-quantization/README.md) | 比较 FP8、W8A8、W4A8 等量化方案，分析权重、激活和 KV Cache 分别如何量化。 |
| 第 10 周第 1 讲 | [量化实践与模型选择](2026-capstone/week10-1-quantization-practice/README.md) · [作业](2026-capstone/week10-1-quantization-practice/assignment.md) | 在 DGX Spark 上比较 Qwen3-8B 的 BF16 与 FP8 版本，根据内存占用、速度和精度选择模型。 |
| 第 10 周第 2 讲 | [智能体规划（Agent Planning）](2026-capstone/week10-2-agent-planning/README.md) | 以 LangGraph 测试报告助手为例，学习如何拆分任务、根据工具结果调整计划，以及在中断后恢复执行。 |
| 第 11 周 | [智能体评估与执行记录（Trace）](2026-capstone/week11-1-agent-eval-trace/README.md) · [作业](2026-capstone/week11-1-agent-eval-trace/assignment.md) | 在同一组任务上评估 Agent v2，比较加入规划前后的成功率、工具调用次数和耗时，沿执行记录定位失败步骤。 |

## 2025 年 · 手机端部署与推理优化

在手机上运行模型可从 Android 教程入手。关心显存容量时，可选读模型量化和协同推理专题。各专题的运行环境和示例见对应目录。[2025 年专题索引](archive/README.md)

| 专题资料 | 学习与实践 | 助教 |
| --- | --- | --- |
| [Android 手机端推理](archive/AndroidLLMInference_Tutorial/Readme.md) | 在 Termux 中配置 Vulkan、编译 `llama.cpp`，并在手机上加载 GGUF 模型进行推理。 | 龙泠锟 |
| [模型量化（Quantization）](archive/Quantization_Tutorial/README.md) | 使用 FP8、INT8 或 GPTQ 示例量化模型，再用 Transformers 或 vLLM 加载模型进行推理。 | 杨濡冰 |
| [CPU/GPU 协同推理（Co-Inference）](archive/Co-Inference_Tutorial/README.md) | 用 FlexGen 调整模型在 GPU、内存和磁盘上的分配比例，对比不同卸载策略的显存占用与推理速度。 | 周傲 |
| [StreamingLLM 长序列推理](archive/Streamingllm_Tutorial/) | 阅读长输入的 KV Cache 实现，观察它如何保留序列开头与最近的 token。配套脚本可记录生成耗时和吞吐量。[示例脚本](archive/Streamingllm_Tutorial/demo.py) | 龙泠锟 |
| [Triton GPU 算子编程](archive/Triton_Tutorial/triton.md) | 用 Triton 编写 Softmax 算子，与 PyTorch 实现对比，并测试替换算子后的模型推理性能。 | 郝泽钰 |
| [扩散语言模型（DLLM）](archive/DLLM_Tutorial/README.md) | 以 LLaDA 为例，研究双向注意力、单步多 token 解码和 KV Cache 复用，并尝试稀疏注意力。 | 龙泠锟 |
