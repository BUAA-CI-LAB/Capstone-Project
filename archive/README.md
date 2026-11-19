# 2025 年课程资料 · 手机端部署与推理优化

2025 年的顶点课程延续了在智能手机上部署大模型及其应用的实践。本页收录六个专题。首次做手机端部署，可以从 Android 教程开始。遇到显存瓶颈时，可以选读模型量化和协同推理，分别考察降低权重精度、把部分数据移到内存或磁盘对推理的影响。

手机端实验在 Android 的 Termux 环境中完成。运行 Python 示例前，请先进入对应的专题目录，按该专题的说明准备独立环境和模型权重。

## 专题与实验入口

| 专题 | 可以研究的问题 | 首读文档与示例 | 助教 |
| --- | --- | --- | --- |
| Android 手机端推理 | 怎样编译 `llama.cpp`，用手机 GPU 运行本地 GGUF 模型？ | [Termux 配置、Vulkan 加速与模型运行](AndroidLLMInference_Tutorial/Readme.md) | 龙泠锟 |
| 模型量化（Quantization） | 如何将同一个模型量化为不同精度，并加载量化后的权重进行推理？ | [量化教程](Quantization_Tutorial/README.md) · [动态 FP8](Quantization_Tutorial/fp8.py) · [W8A8 INT8](Quantization_Tutorial/int8.py) · [W4A8 GPTQ](Quantization_Tutorial/w4a8_gptq.py) | 杨濡冰 |
| CPU/GPU 协同推理（Co-Inference） | 单卡显存有限时，改变权重和缓存的卸载比例会如何影响显存占用与推理速度？ | [FlexGen 教程](Co-Inference_Tutorial/README.md) · [纯 GPU 基线](Co-Inference_Tutorial/onlygpu_demo.py) · [卸载对比实验](Co-Inference_Tutorial/flexgen_demo.py) | 周傲 |
| StreamingLLM 长序列推理 | 为长输入保留序列开头和最近 token 的 KV 缓存，会如何影响生成耗时与吞吐量？ | [专题目录](Streamingllm_Tutorial/) · [长输入测试脚本](Streamingllm_Tutorial/demo.py) · [依赖清单](Streamingllm_Tutorial/requirements.txt) | 龙泠锟 |
| Triton GPU 算子编程 | 如何编写 Softmax 算子，并用 PyTorch 的实现核对数值结果、比较执行耗时？ | [Triton 教程](Triton_Tutorial/triton.md) · [Softmax 性能对比](Triton_Tutorial/comparison.py) · [模型集成示例](Triton_Tutorial/triton_example.py) | 郝泽钰 |
| 扩散语言模型（DLLM） | LLaDA 如何在一个解码步骤生成多个 token？双向注意力与 KV 缓存如何配合？ | [推理原理与课后实践](DLLM_Tutorial/README.md) · [生成示例](DLLM_Tutorial/llada/generate.py) · [评测指南](DLLM_Tutorial/llada/eval.md) | 龙泠锟 |

## 运行提示

- **模型量化**　表中的三个量化脚本约定 `MODEL_DIR` 指向包含 `Qwen3-0.6B/` 的父目录。各脚本将对应的量化结果写入该父目录下的独立子目录。
- **StreamingLLM**　先将 `demo.py` 中的 `model_name` 和 `device` 设为本机模型路径和 GPU 编号，再从专题目录启动脚本，以便读取 `long_prompt.txt`。脚本默认测量标准注意力基线，做对照时可启用脚本末尾的 StreamingLLM 示例，复用同一份输入。
- **LLaDA**　运行 `llada/generate.py` 前，在 `main()` 中将模型和分词器的路径指向本机权重目录，并设置可用的 CUDA 设备。

[返回课程首页](../README.md)
