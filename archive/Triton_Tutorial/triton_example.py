#!/usr/bin/env python3
"""
Triton Softmax 算子实现与 Transformers 集成示例
"""

import torch
import torch.nn as nn
import time
import triton
import triton.language as tl

# ============================================================================
# 1. Triton Softmax 算子实现
# ============================================================================

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols

    row = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    exp_row = tl.exp(row_minus_max)
    softmax_row = exp_row / tl.sum(exp_row, axis=0)

    tl.store(output_ptr + offsets, softmax_row, mask=mask)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    softmax_kernel[grid](
        output, x, n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

# ============================================================================
# 2. 算子测试
# ============================================================================

def test_softmax():
    print("=== Triton Softmax 测试 ===")

    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA 设备")
        return

    torch.manual_seed(42)
    x = torch.randn(1024, 4096, device='cuda')
    print(f"测试数据形状: {x.shape}")

    # Warmup
    print("进行 warmup...")
    _ = triton_softmax(x)
    torch.cuda.synchronize()
    _ = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # 性能测试
    test_runs = 50

    # PyTorch Softmax
    start = time.time()
    for _ in range(test_runs):
        y_torch = torch.softmax(x, dim=1)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / test_runs

    # Triton Softmax
    start = time.time()
    for _ in range(test_runs):
        y_triton = triton_softmax(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / test_runs

    # 验证正确性
    max_diff = torch.max(torch.abs(y_torch - y_triton)).item()
    print(f"最大差异: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("✓ 正确性测试通过")
    else:
        print("✗ 正确性测试失败")
        return

    # 性能比较
    print(f"PyTorch Softmax: {torch_time*1000:.3f}ms")
    print(f"Triton Softmax:  {triton_time*1000:.3f}ms")
    print(f"加速比: {torch_time/triton_time:.2f}x")

# ============================================================================
# 3. Transformers 集成
# ============================================================================

original_softmax = torch.nn.functional.softmax

def patched_softmax(x, dim=None, _stacklevel=3, dtype=None):
    if dim is not None and dim != -1 and dim != x.dim() - 1:
        dims = list(range(x.dim()))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        x = x.permute(dims)
        transposed = True
    else:
        transposed = False

    if x.dim() > 2:
        original_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        result_2d = triton_softmax(x_2d)
        result = result_2d.view(original_shape)
    else:
        result = triton_softmax(x)

    if transposed:
        result = result.permute(dims)

    return result

def replace_softmax_in_model():
    torch.nn.functional.softmax = patched_softmax

def restore_original_softmax():
    torch.nn.functional.softmax = original_softmax

# ============================================================================
# 4. 模型性能测试
# ============================================================================

def test_model_integration():
    print("\n=== 模型集成测试 ===")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("警告: 未安装 transformers 库")
        print("运行: pip install transformers")
        return

    model_name = "gpt2"
    print(f"加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").cuda()
    model.eval()

    test_prompt = "Hello, world!"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    gen_kwargs = {
        "max_length": 20,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False
    }

    # 原始模型 warmup
    print("\n原始模型 warmup...")
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()

    # 原始模型测试
    print("原始模型测试...")
    test_runs = 10
    with torch.no_grad():
        start = time.time()
        for _ in range(test_runs):
            output1 = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        original_time = (time.time() - start) / test_runs

    # 替换 softmax
    print("\n替换为 Triton Softmax...")
    replace_softmax_in_model()

    # 优化模型 warmup
    print("优化模型 warmup...")
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()

    # 优化模型测试
    print("优化模型测试...")
    with torch.no_grad():
        start = time.time()
        for _ in range(test_runs):
            output2 = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        optimized_time = (time.time() - start) / test_runs

    # 结果比较
    print(f"\n原始模型时间: {original_time*1000:.2f}ms")
    print(f"优化模型时间: {optimized_time*1000:.2f}ms")
    print(f"加速比: {original_time/optimized_time:.2f}x")

    # 检查输出一致性
    print("\n生成的文本:")
    print(f"原始: {tokenizer.decode(output1[0], skip_special_tokens=True)}")
    print(f"优化: {tokenizer.decode(output2[0], skip_special_tokens=True)}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("Triton 实践教程")
    print("=" * 40)

    if not torch.cuda.is_available():
        print("需要 CUDA 设备才能运行此示例")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Triton 版本: {triton.__version__}")

    # 运行测试
    test_softmax()
    test_model_integration()

    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()