# Triton GPU 编程理论基础

## Triton 简介

Triton 是 OpenAI 开发的 Python 编译器和语言，用于编写高性能的自定义深度学习算子。它提供比 CUDA 更简单、更高效的 GPU 编程方式。

## 编程模型对比

### Triton vs CUDA

| 特性 | CUDA | Triton |
|------|------|--------|
| **编程抽象** | 线程级 | 块级 |
| **内存管理** | 手动优化 | 自动优化 |
| **代码复杂度** | 高 | 低 |
| **开发效率** | 较低 | 较高 |
| **学习曲线** | 陡峭 | 平缓 |

### 块级编程模型

**CUDA 线程级编程**:
```cpp
#pragma parallel
for(int m = 0; m < M; m++) {
    #pragma parallel
    for(int n = 0; n < N; n++) {
        float acc = 0;
        for(int k = 0; k < K; k++)
            acc += A[m, k] * B[k, n];
        C[m, n] = acc;
    }
}
```

**Triton 块级编程**:

```cpp
#pragma parallel
for(int m = 0; m < M; m += MB) {
    #pragma parallel
    for(int n = 0; n < N; n += NB) {
        float acc[MB, NB] = 0;
        for(int k = 0; k < K; k += KB)
            acc += A[m:m+MB, k:k+KB] @ B[k:k+KB, n:n+NB];
        C[m:m+MB, n:n+NB] = acc;
    }
}
```

## 核心 API

### 基础函数

- **`@triton.jit`**: 将 Python 函数编译为 GPU 内核
- **`tl.program_id(axis)`**: 获取当前程序实例的 ID
- **`tl.arange(start, stop)`**: 创建范围张量
- **`tl.load(pointer, mask=None)`**: 从内存加载数据
- **`tl.store(pointer, value, mask=None)`**: 将数据存储到内存
- **`tl.constexpr`**: 编译时常量标记

### 内存管理

```python
# 创建掩码防止越界访问
mask = offsets < n_elements
# 加载数据时使用掩码
data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
```

## 系统要求

- **操作系统**: Linux (推荐 Ubuntu 22.04+) / Windows（需要 WSL）
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+
- **CUDA**: CUDA 11.0+
- **Python**: Python 3.8-3.12

## 安装方法

### 方法一：pip 安装（推荐）

```bash
# 安装最新稳定版
pip install triton

# 安装 nightly 版本
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

### 方法二：从源码安装

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
pip install ninja cmake wheel
cd python
pip install -e .
```

### 验证安装

```python
import torch
import triton

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Triton version: {triton.__version__}")
```

## Softmax 算子实现

### 数学定义

Softmax 的数学公式：
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}$$

### Triton 实现

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # 获取当前处理的行号
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols

    # 加载当前行数据
    row = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))

    # 数值稳定的 softmax
    row_minus_max = row - tl.max(row, axis=0)
    exp_row = tl.exp(row_minus_max)
    softmax_row = exp_row / tl.sum(exp_row, axis=0)

    # 存储结果
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
```

## 关键概念

### 1. Warmup

Triton 内核首次运行时需要编译，首次执行会比后续执行慢很多倍。

```python
# 进行 warmup
for _ in range(5):
    _ = triton_softmax(data)
torch.cuda.synchronize()

# 现在可以测量真实性能
```

### 2. 边界检查

使用掩码防止越界访问：
```python
mask = offsets < row_start + n_cols
data = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
```

## 模型集成方法

### Monkey Patch 技术

通过替换 PyTorch 的 softmax 函数来集成自定义算子：

```python
import torch.nn.functional as F

# 保存原始函数
original_softmax = F.softmax

# 定义替换函数
def patched_softmax(x, dim=None, **kwargs):
    # 处理维度转换
    if dim != -1 and dim != x.dim() - 1:
        dims = list(range(x.dim()))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        x = x.permute(dims)
        transposed = True
    else:
        transposed = False

    # 调用 Triton 实现
    result = triton_softmax(x)

    # 恢复原始维度顺序
    if transposed:
        result = result.permute(dims)

    return result

# 应用替换
F.softmax = patched_softmax
```

## 性能优化策略

### 1. 内存访问模式
- 确保连续内存访问
- 使用合适的块大小
- 利用缓存局部性

### 2. 算子融合
- 减少内存访问次数
- 降低内存带宽需求
- 提高计算密度

### 3. 自动调优
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['n_cols']
)
def autotune_softmax_kernel(...):
    # 内核实现
    pass
```

## 应用场景

### 适用场景
- 自定义激活函数
- 复杂的数据处理流水线
- 没有标准库支持的操作
- 需要算子融合的场景

### 不适用场景
- 已经有高度优化的标准库实现（如 cuDNN）
- 简单的元素级操作

## 参考资料

- [Triton 官方文档](https://triton-lang.org/)
- [GitHub 仓库](https://github.com/triton-lang/triton)
- [学术论文](https://arxiv.org/abs/2101.06802)
- [示例代码](https://github.com/triton-lang/triton/tree/main/python/tutorials)