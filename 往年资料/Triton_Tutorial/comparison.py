#!/usr/bin/env python3
"""
Triton vs PyTorch Softmax 性能对比实验
"""

import torch
import time
import triton
import triton.language as tl

# ============================================================================
# 1. Triton Softmax 实现
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
# 2. 手动 PyTorch Softmax 实现
# ============================================================================

def manual_softmax_torch(x):
    x_max = torch.max(x, dim=1, keepdim=True)[0]    # load + compute + store
    x_exp = torch.exp(x - x_max)                    # load + compute + store
    return x_exp / torch.sum(x_exp, dim=1, keepdim=True)    # (load + compute + store) * 2

# ============================================================================
# 3. 性能测试框架
# ============================================================================

def benchmark_function(func, x, warmup_runs=50, test_runs=100):
    for _ in range(warmup_runs):
        _ = func(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(test_runs):
        result = func(x)
    torch.cuda.synchronize()
    avg_time = (time.time() - start) / test_runs

    return result, avg_time

def run_all_tests():
    print("正在运行性能测试...")

    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA 设备")
        return None

    test_sizes = [
        (1024, 1024),
        (2048, 2048),
        (4096, 1024),
        (1024, 4096)
    ]

    device = torch.device('cuda')
    torch.manual_seed(42)
    results = []

    for i, size in enumerate(test_sizes, 1):
        print(f"测试进度: {i}/{len(test_sizes)} - 大小: {size}")

        x = torch.randn(size, device=device)

        triton_result, triton_time = benchmark_function(triton_softmax, x)
        manual_result, manual_time = benchmark_function(manual_softmax_torch, x)
        pytorch_result, pytorch_time = benchmark_function(
            lambda x: torch.softmax(x, dim=1), x
        )

        reference = torch.softmax(x, dim=1)
        triton_diff = torch.max(torch.abs(reference - triton_result)).item()
        manual_diff = torch.max(torch.abs(reference - manual_result)).item()

        triton_speedup = manual_time / triton_time if triton_time > 0 else 0
        pytorch_speedup = manual_time / pytorch_time if pytorch_time > 0 else 0

        results.append({
            'size': size,
            'triton_time': triton_time * 1000,
            'manual_time': manual_time * 1000,
            'pytorch_time': pytorch_time * 1000,
            'triton_speedup': triton_speedup,
            'pytorch_speedup': pytorch_speedup,
            'triton_correct': triton_diff < 1e-4,
            'manual_correct': manual_diff < 1e-4
        })

    return results

def print_results_summary(results):
    if not results:
        return

    print("\n" + "=" * 60)
    print("性能测试结果汇总")
    print("=" * 60)

    print(f"{'矩阵大小':12} {'Triton':12} {'手动PyTorch':12} {'PyTorch原生':12} {'Triton加速':12}")
    print("-" * 70)

    triton_speedups = []
    for result in results:
        size_str = f"{result['size'][0]}x{result['size'][1]}"
        triton_time = result['triton_time']
        manual_time = result['manual_time']
        pytorch_time = result['pytorch_time']
        speedup = result['triton_speedup']

        print(f"{size_str:12} {triton_time:8.4f}ms {manual_time:10.4f}ms {pytorch_time:10.4f}ms {speedup:10.2f}x")
        triton_speedups.append(speedup)

    print("-" * 60)

    avg_speedup = sum(triton_speedups) / len(triton_speedups)
    max_speedup = max(triton_speedups)
    min_speedup = min(triton_speedups)

    print(f"\n📊 性能统计:")
    print(f"  平均加速比: {avg_speedup:.3f}x")
    print(f"  最大加速比: {max_speedup:.3f}x")
    print(f"  最小加速比: {min_speedup:.3f}x")

    all_correct = all(r['triton_correct'] and r['manual_correct'] for r in results)
    print(f"\n✅ 正确性验证: {'全部通过' if all_correct else '存在问题'}")

def main():
    print("Triton vs PyTorch Softmax 性能对比")
    print("=" * 40)

    if not torch.cuda.is_available():
        print("需要 CUDA 设备才能运行此测试")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Triton 版本: {triton.__version__}")

    results = run_all_tests()
    print_results_summary(results)

    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()