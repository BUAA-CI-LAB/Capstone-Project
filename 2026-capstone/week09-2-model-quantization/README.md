# 第九周第 2 次课：模型量化技术

- [课程课件](model-quantization.pptx)

本次课程讨论模型本身的压缩：从权重与 KV Cache 的显存构成出发，介绍 W8 / W4 量化、权重量化与激活量化的区别，以及 FP8、W8A8 + SmoothQuant、W4A8 GPTQ 等常用方案和 KV Cache 量化。

## 课程内容

1. **显存占用与量化动机**：权重与 KV Cache 的显存构成，比特数、压缩比与精度风险。
2. **量化的分类与作用范围**：按精度、阶段、对象与粒度划分量化方案，权重量化与激活量化的区别，Decoder Block 中被量化的矩阵乘与保持原精度的部分，缩放与取整、scale 的共享范围与确定时机，以及 WxAy 命名。
3. **四种主流方法**：FP8 Dynamic、FP8 分块量化、W8A8 + SmoothQuant、W4A8 GPTQ，并与 AWQ 对比。
4. **KV Cache 量化**：`--kv-cache-dtype` 与 scale 标定，量化粒度与注意力后端，FP8 KV Cache 的实测结果与适用场景。

## 资料拓展

课外参考资料（llmcompressor 文档、vLLM Quantized KV Cache、GPTQ / AWQ / SmoothQuant 论文、DeepSeek-V3 技术报告等）见课件最后一页。

[返回课程目录](../README.md)
