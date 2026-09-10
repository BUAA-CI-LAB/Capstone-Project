# Quantization Tutorial

## 快速上手

### 环境配置

建议环境： `python>=3.9`。

通过 pip 安装 `llmcompressor`：
```bash
pip install llmcompressor
```

`llmcompressor` 是一个强大的量化工具，其源码可以从 [llm-compressor](https://github.com/vllm-project/llm-compressor) 获取。

### 示例代码

我们提供了几种常见的量化方案的示例代码：

- **FP8 动态量化 (FP8 Dynamic Quantization):**
    ```shell
    MODEL_DIR=YOUR_PATH python3 fp8_dynamic.py
    ```
    这种方法将模型的权重和激活值动态地量化为 8 位浮点数 (FP8)，可以在基本不损失模型精度的情况下，显著提升推理速度和降低显存占用。

    Block-wise动态量化：

    ```shell
    python3 fp8_blockwise --input_path INPUT_PATH --output_path OUTPUT_PATH
    ```

- **W8A8 对称量化 (W8A8 Symmetric Quantization):**
    ```shell
    MODEL_DIR=YOUR_PATH python3 w8a8.py
    ```
    此方法将权重和激活值量化为 8 位整数 (INT8)。为了减少量化带来的精度损失，我们通常会结合 SmoothQuant 技术。

- **GPTQ W4A8 量化 (GPTQ W4A8 Quantization):**
    ```shell
    MODEL_DIR=YOUR_PATH python3 w4a8_gptq.py
    ```
    GPTQ 是一种训练后量化方法，它通过对权重进行逐列的量化来减少精度损失。在这里，我们将权重 `W` 量化为 4 位，激活值 `A` 量化为 8 位。

- **Transformers 离线伪量化推理 (Transformers Offline Pseudo-quantized Inference):**
    ```shell
    python3 transformers_offline.py $MODEL_PATH
    ```
    这个脚本展示了如何在 `transformers` 框架中加载一个已经伪量化好的模型，并进行推理。

- **vLLM 在线推理 (vLLM Online Inference):**
    ```shell
    vllm serve $MODEL_PATH
    bash openai.sh
    ```
    对于需要高性能在线服务的场景，`vLLM` 是一个非常好的选择。它能够高效地利用显存，并提供 OpenAI 兼容的 API 接口。

## 核心代码解析

### 模型结构

通常，量化会应用在模型的线性层（如 `torch.nn.Linear`）上。`llm-compressor` 使用 `ModuleHolder` 来包装这些层，并通过 Modifier 来修改它们的前向传播行为。

例如，Llama的模型结构如下：
```python
LlamaDecoderLayer(
  (self_attn): LlamaAttention(
    (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
    (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
    (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (mlp): LlamaMLP(
    (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
    (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
    (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
    (act_fn): SiLU()
  )
  (input_layernorm): LlamaRMSNorm()
  (post_attention_layernorm): LlamaRMSNorm()
)
```

### FP8量化


FP8（8位浮点数）量化是一种旨在保留动态范围和精度的同时大幅减少模型大小和计算量的技术。与INT8不同，FP8使用浮点格式（符号、指数、尾数），使其能更好地处理大模型中常见的异常值（outliers）。

-   **量化 (Quantization)**
    在 `llm-compressor` 中，FP8量化是通过 `QuantizationModifier` 实现的。当你在配置文件中指定 `num_bits: 8` 和 `type: "float"` 时，就会启用FP8量化。

    **代码位置**: `src/llmcompressor/modifiers/quantization/quant_modifier.py`

    该文件中的 `QuantizationModifier` 类会解析用户的配置，并为模型的指定层（如 `Linear`）应用量化。它会创建量化方案（Scheme）和包装层（Wrapper），在模型前向传播时动态地对权重和激活值进行量化和反量化。

-   **伪量化 (Fake Quantization)**
    在训练或校准（calibration）过程中，我们通常使用“伪量化”。这意味着权重和激活值被量化到FP8，然后立即被反量化回FP32进行计算。这个过程模拟了量化带来的精度损失，使得模型能够适应这种损失，但计算本身仍在FP32上进行。

```python
def pseudo_quantize_tensor(
    w, w_bit=4, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**w_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (w_bit - 1) - 1
        min_int = -(2 ** (w_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
```


### INT8量化

INT8（8位整数）量化是最常见和广泛支持的量化类型之一。它将FP32的权重和激活值映射到[-128, 127]或[0, 255]的整数范围内。

-   **量化**
    `llm-compressor` 使用与FP8相同的框架来处理INT8量化。`QuantizationModifier` 会根据配置文件中的 `type: "int"` 来应用INT8量化方案。底层的伪量化逻辑也由 `apply_fake_quant` 函数处理，只是量化范围（`quant_min`, `quant_max`）和数据类型不同。


- **伪量化**
```python
def fake_quant_dequant(x, method="abs_max", bits=8, group_size=-1):
    bnt = (1 << (bits - 1)) - 1
    quant_scale = compute_scales(x, method=method, group_size=group_size)
    if method == "groupwise":
        quant_scale = torch.repeat_interleave(quant_scale, group_size, dim=-1)
        quant_scale = quant_scale.reshape(x.shape)
    for _ in range(len(x.shape) - len(quant_scale.shape)):
        quant_scale = quant_scale.unsqueeze(-1)
    quant_value = torch.clamp(torch.round(x / quant_scale * bnt), -bnt - 1, bnt)
    quant_dequant_value = quant_value / bnt * quant_scale
    return quant_dequant_value
```


### SmoothQuant

SmoothQuant是一种解决大语言模型量化难题的技术。在LLM中，激活值的分布通常非常不均衡，存在一些具有巨大数值的“异常值”，这使得INT8量化变得困难。SmoothQuant通过将一部分量化难度从激活值（Activations）“平滑”到权重（Weights）上，来缓解这个问题。它对每个通道应用一个缩放因子，降低激活值的动态范围，同时相应地调整权重的值，从而在不改变数学计算结果的前提下，让模型变得更容易量化。

**代码位置**: `src/llmcompressor/modifiers/smoothquant/base.py`

`SmoothQuantModifier` 类负责执行这个过程。它会在量化校准（calibration）阶段运行：
1.  首先，它会观察模型在一小部分校准数据上的激活值，以计算出合适的“平滑”缩放因子。
2.  然后，它会修改模型的权重，将缩放因子“吸收”进去。
3.  最后，应用标准的INT8量化。

```python name=src/llmcompressor/modifiers/smoothquant/smoothquant.py url=https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/smoothquant/smoothquant.py
class SmoothQuantModifier(QuantizationModifier):
    # ... (initialization)

    def _apply_smoothing(self, module_holder: ModuleHolder, state: State):
        # ...
        for name, submodule in module_holder.module.named_modules():
            if not self._smoothing_hook_applied(submodule):
                # find all linear layers to smooth
                if isinstance(submodule, self.get_qat_module_type(config)):
                    # ...
                    # register forward hooks to gather activation statistics
                    handle = submodule.register_forward_pre_hook(
                        self.calibration_hook(
                            scales=scales,
                            submodule_name=f"{module_holder.name}.{name}",
                            inplace=self.inplace,
                        )
                    )
        # ...
        # run calibration data through the model
        self.run_calibration_forward(state.data.calib)
        
        # ...
        # apply smoothing scales to the weights
        self._apply_smoothing_scales(module_holder.module, scales)

    # ...
    @torch.no_grad()
    def _apply_smoothing_scales(self, module: torch.nn.Module, scales: Dict):
        for name, submodule in module.named_modules():
            if name in scales:
                # ...
                # scale the weights of the smoothable linear layer
                submodule.weight.div_(
                    scales[name].reshape((1, -1)).to(submodule.weight.device)
                )

                # scale the bias if it exists
                if hasattr(submodule, "bias") and submodule.bias is not None:
                    submodule.bias.div_(
                        scales[name].reshape((-1)).to(submodule.bias.device)
                    )
```

### GPTQ

GPTQ (Generative Pre-trained Transformer Quantization) 是一种先进的训练后量化（Post-Training Quantization, PTQ）方法。它通过一种基于量纲的误差补偿策略，能够将模型的权重压缩到极低的位数（如 4 位），同时保持较高的模型性能。

`GPTQModifier` 封装了 GPTQ 的量化逻辑。其核心思想可以概括为：

1.  **逐列量化**：GPTQ 独立地量化权重矩阵的每一列。
2.  **误差补偿**：当一列中的某个权重被量化后，产生的误差会根据 Hessian 矩阵的逆来更新该行中所有剩余未被量化的权重，从而补偿这个误差。

下面我们深入到代码层面，看看它是如何实现的。

#### 核心代码解析

GPTQ 的主要实现在 `src/llmcompressor/modifiers/quantization/gptq/core.py` 中。

##### 1. Hessian 矩阵的逆的计算

GPTQ 算法的关键一步是计算 Hessian 矩阵的逆，并用它来指导权重的更新。在 `GPTQ` 类的 `_add_batch` 方法中，Hessian 矩阵 `H` 是通过输入 `inp` 的外积累加得到的。

Hessian矩阵会用于后面逐层量化过程中的损失和补偿计算，所以需要先离线计算得到。实现方式是在每一层Layer上注册hook，通过hook的方式在layer forward后使用calibration data的input来生成Hessian矩阵，这种计算方式常见于量化流程中校准数据的处理。

```python
def add_batch(self, inp, out):
    # Hessian H = 2 X XT + λ I
    if self.observe:
        self.inp1 = inp
        self.out1 = out
    else:
        self.inp1 = None
        self.out1 = None

    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    tmp = inp.shape[0]
    if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
    if isinstance(self.layer, nn.Conv2d):
        unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride)
        inp = unfold(inp)
        inp = inp.permute([1, 0, 2])
        inp = inp.flatten(1)
    # 核心就是下面几行
    self.H *= self.nsamples / (self.nsamples + tmp)
    self.nsamples += tmp
    inp = math.sqrt(2 / self.nsamples) * inp.float()
    self.H += inp.matmul(inp.t())
```

##### 2. 权重迭代量化与更新

在 `_quantize_column` 方法中，GPTQ 逐列对权重进行量化。对于每一列，它会：
1.  计算量化后的权重 `q` 和产生的误差 `err`。
2.  使用 Hessian 矩阵的逆 `H_inv` 来更新剩余的权重 `W`，将误差分散出去。
3.  更新 Hessian 矩阵的逆。

这个过程不断重复，直到所有权重都被量化。

```python
    def _quantize_column(
        self, W: torch.Tensor, H_inv: torch.Tensor, perm: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ...
        for i in range(self.columns):
            # ...
            w = W[:, i]
            q = self._quantize_val(w)
            Q[:, i] = q
            err = w - q
            # ...
            W[:, i:] -= err.unsqueeze(1).matmul(H_inv[i, i:].unsqueeze(0))
            # ...
        return Q, err_norm / self.columns
```

通过这种方式，GPTQ 巧妙地在量化过程中保持了模型的性能，使其成为一种非常流行和有效的低比特量化方案。

### AWQ

核心思想：用激活值来发现重要weight。对weight进行per-channel的scale同时对激活值除以scale来保护重要weight。取和激活值相关的值进行grid search，找到那个让量化误差最小的scale

逐层地去计算需要调整的weight，每一层的输出会作为下一层的输入，在LlamaDecoderLayer内部使用hook的方式来记录每一线性子层的input_feature，和GPTQ的做法类似。

参考：https://github.com/mit-han-lab/llm-awq

```python
def _search_module_scale(block, linears2scale: list, x, kwargs={}):
    # w: co, ci
    # x: n, ci
    x = x.to(next(block.parameters()).device)
    with torch.no_grad():
        org_out = block(x, **kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]
    x_max = get_act_scale(x)
    best_error = float("inf")
    best_ratio = -1
    best_scales = None

    n_grid = 20
    history = []

    org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
    for ratio in range(n_grid):
        ratio = ratio * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()

        for fc in linears2scale:
            fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
            fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
        out = block(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]

        loss = (
            (org_out - out).float().pow(2).mean().item()
        )  # float prevents overflow
        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_ratio = ratio
            best_scales = scales
        block.load_state_dict(org_sd)
    if best_ratio == -1:
        print(history)
        raise Exception
    # print(best_ratio)
    best_scales = best_scales.view(-1)

    assert torch.isnan(best_scales).sum() == 0, best_scales
    return best_scales.detach()
```


## 拓展学习

除了上述介绍的方法，量化领域还有许多其他值得探索的方向：

- **细粒度量化**：
    比较NVFP4和MXFP4的区别，为什么block对应尾数位为0，怎么在hooper架构上运行FP4算子
- **旋转矩阵低比特量化:** 
    处理激活异常值的方法，例如 quarot、spinquant等
- **GPTAQ:**
    逐层量化时直接追随原值，非对称校准
- **QAT (Quantization-Aware Training):** 在模型训练过程中就引入量化操作，让模型自己去适应量化带来的噪声，从而获得比 PTQ 更高的精度。