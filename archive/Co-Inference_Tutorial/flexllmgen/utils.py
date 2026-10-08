import argparse  
import dataclasses  
from attr import define, field  
from attr.setters import frozen  
import functools  
import gc  
import math  
import os 
from typing import Tuple, Union, Optional, Any, Sequence, List 

import numpy as np 
import torch  


# 字节单位转换常量（二进制）
KB = 1 << 10  
MB = 1 << 20  
GB = 1 << 30  
T = 1e12  # （十进制，用于粗略计算）


@dataclasses.dataclass(frozen=True)
class Task:
    """生成任务的数据类（不可变）。"""
    inputs: Union[np.array, List[List[int]]]  # 输入数据，可为numpy数组或整数列表的列表（token序列）
    prompt_len: int  # 提示文本的长度（token数）
    gen_len: int  # 要生成的文本长度（token数）
    cut_gen_len: Optional[int]  # 截断生成长度（可选，用于限制实际生成长度）

    do_sample: bool  # 是否使用采样策略生成（而非贪婪解码）
    temperature: float  # 采样温度（控制随机性，值越高越随机）
    stop: Optional[int]  # 终止token（遇到该token时停止生成，可选）


@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """硬件执行环境的数据类（不可变）。"""
    gpu: Any = None  # GPU设备对象
    cpu: Any = None  # CPU设备对象
    disk: Any = None  # 磁盘存储对象
    mixed: Any = None  # 混合设备（GPU+CPU+磁盘）对象

    @classmethod
    def create(cls, offload_dir):
        """创建硬件环境实例。"""
        # 解决循环导入问题
        from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
        gpu = TorchDevice("cuda:0")  # 初始化GPU设备（默认第0块GPU）
        cpu = TorchDevice("cpu")  # 初始化CPU设备
        disk = TorchDisk(offload_dir)  # 初始化磁盘存储（指定卸载目录）
        return cls(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))  # 创建混合设备环境

    def close_copy_threads(self):
        """关闭磁盘的复制线程（释放资源）。"""
        self.disk.close_copy_threads()


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """基准测试结果的数据类（不可变）。"""
    prefill_latency: float  # 预填充阶段延迟（秒）
    prefill_throughput: float  # 预填充阶段吞吐量（token/秒）
    decode_latency: float  # 解码阶段延迟（秒）
    decode_throughput: float  # 解码阶段吞吐量（token/秒）
    total_latency: float  # 总延迟（秒）
    total_throughput: float  # 总吞吐量（token/秒）


# 数据类型转换映射：numpy dtype -> torch dtype
np_dtype_to_torch_dtype = {
    np.float16: torch.float16, np.float32: torch.float32, np.uint8: torch.uint8,
    np.int8: torch.int8, np.int32: torch.int32, np.int64: torch.int64,
    bool: torch.bool,
}

# 数据类型转换映射：torch dtype -> numpy dtype
torch_dtype_to_np_dtype = {
    torch.float16: np.float16, torch.float32: np.float32,
    torch.uint8: np.uint8, torch.int8: np.int8, torch.int32: np.int32,
    torch.int64: np.int64, torch.bool: bool,
}

# 数据类型字节数映射：torch dtype -> 单个元素字节数
torch_dtype_to_num_bytes = {
    torch.float16: 2, torch.float32: 4,
    torch.int8: 1, torch.uint8: 1, torch.int32: 4, torch.int64: 8,
    torch.bool: 1,
}


def piecewise_linear_func(xs, ys):
    """创建一个分段线性插值函数。"""
    indices = np.argsort(xs)  # 对x坐标排序
    xs = [xs[i] for i in indices]
    ys = [ys[i] for i in indices]

    # 扩展左右边界（避免插值超出范围）
    k = 1e5
    delta_x_left = xs[0] - xs[1]
    delta_y_left = ys[0] - ys[1]
    delta_x_right = xs[-1] - xs[-2]
    delta_y_right = ys[-1] - ys[-2]

    xs = [xs[0] + delta_x_left * k] + xs + [xs[-1] + delta_x_right * k]
    ys = [ys[0] + delta_y_left * k] + ys + [ys[-1] + delta_y_right * k]

    # 返回绑定了xs和ys的偏函数
    return functools.partial(piecewise_linear_func_ret_func, xs, ys)


def piecewise_linear_func_ret_func(xs, ys, x):
    """分段线性插值函数的具体实现（被piecewise_linear_func调用）。"""
    assert x >= xs[0] and x <= xs[-1]  # 确保x在插值范围内
    return np.interp(x, xs, ys)  # 使用numpy的线性插值


def sample_from_range(n, k):
    """从范围[1, n]中采样k个值（用于生成测试用例）。"""
    assert n >= 1  # 确保范围有效

    if k == -1:
        # 特殊情况：生成以2为倍数的序列（1, 2, 4, ... 直到小于n）
        ret = [1]
        while ret[-1] * 2 < n:
            ret.append(ret[-1] * 2)
        return ret
    else:
        if k == 1:
            return [1]  # 仅返回1
        # 均匀采样k个值（步长为(n-1)/(k-1)）
        step = (n - 1) // (k - 1)
        return list(range(1, n + 1, step))


def cpu_mem_stats():
    """统计CPU上所有PyTorch张量的总内存占用（字节）。"""
    objects = gc.get_objects()  # 获取所有被垃圾回收跟踪的对象
    # 筛选出CPU上的张量（非CUDA张量）
    tensors = [obj for obj in objects if torch.is_tensor(obj) and not obj.is_cuda]

    total_mem = 0
    visited_data = set()  # 用于去重（避免重复计算共享内存的张量）
    for tensor in tensors:
        # 通过数据指针判断是否为同一内存块
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        numel = tensor.numel()  # 张量元素总数
        element_size = tensor.storage().element_size()  # 单个元素字节数
        mem = numel * element_size  # 该张量占用内存
        total_mem += mem

    return total_mem


def torch_mem_stats():
    """统计GPU上所有PyTorch张量的总内存占用（字节），并打印张量信息。"""
    objects = gc.get_objects()  # 获取所有被垃圾回收跟踪的对象
    # 筛选出GPU上的张量（CUDA张量）
    tensors = [obj for obj in objects if torch.is_tensor(obj) and obj.is_cuda]

    total_mem = 0
    visited_data = set()  # 用于去重
    for tensor in tensors:
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        print(tensor.shape, tensor.data_ptr())  # 打印张量形状和数据指针

        numel = tensor.numel()
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem


class ValueHolder:
    """轻量级值容器，用于安全地存储和传递临时数据（如权重、缓存）。"""
    def __init__(self):
        self.val = None  # 存储的值

    def store(self, val):
        """存储值（确保当前为空）。"""
        assert self.val is None
        self.val = val

    def pop(self):
        """取出值并清空（原子操作，避免数据残留）。"""
        ret = self.val
        self.val = None
        return ret

    def clear(self):
        """清空存储的值。"""
        self.val = None


def array_1d(a, cls):
    """创建1维数组，元素为cls类的实例。"""
    return [cls() for _ in range(a)]


def array_2d(a, b, cls):
    """创建2维数组（a行b列），元素为cls类的实例。"""
    return [[cls() for _ in range(b)] for _ in range(a)]


def array_3d(a, b, c, cls):
    """创建3维数组（a×b×c），元素为cls类的实例。"""
    return [[[cls() for _ in range(c)] for _ in range(b)] for _ in range(a)]


def array_4d(a, b, c, d, cls):
    """创建4维数组（a×b×c×d），元素为cls类的实例。"""
    return [[[[cls() for _ in range(d)] for _ in range(c)] for _ in range(b)] for _ in range(a)]


def vector_gather(vectors, indices):
    """
    根据索引收集批量向量（用于注意力机制中的查询）。
    参数：
        vectors: 形状为[S, B, H]的张量（S: 序列长度，B: 批量大小，H: 隐藏层维度）
        indices: 形状为[K, B]的张量（K: 要收集的向量数，B: 批量大小）
    返回：
        形状为[K, B, H]的张量（收集后的向量）
    """
    S, B, H = vectors.shape
    K, B2 = indices.shape
    assert B == B2  # 确保批量大小一致
    # 将indices扩展为[K, B, H]以匹配vectors的维度
    indices = indices.reshape(K, B, 1).expand(K, B, H)
    # 按dim=0（序列维度）收集向量
    out = vectors.gather(dim=0, index=indices)
    return out


def run_cmd(cmd):
    """执行系统命令并打印命令内容。"""
    print(cmd)
    os.system(cmd)


def str2bool(v):
    """将字符串转换为布尔值（处理常见的真/假表示）。"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('需要布尔值（yes/no, true/false, 1/0等）')


def project_decode_latency(costs, prompt_len, gen_len):
    """
    预测解码阶段的总延迟（基于已有成本数据）。
    参数：
        costs: 各步骤的耗时列表（第0项为预填充耗时，后续为解码步骤耗时）
        prompt_len: 提示长度
        gen_len: 生成长度
    返回：
        预测的解码总延迟（秒）
    """
    decode_costs = costs[1:]  # 解码步骤的耗时（排除预填充）

    # 根据生成长度与提示长度的比例调整预热步骤
    if gen_len / prompt_len < 0.1:
        warmup = 2  # 预热步骤数（前2步可能耗时较高）
        # 总延迟 = 预热步骤耗时 + 剩余步骤的平均耗时
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))
    else:
        warmup = 2
        decode_latency = (sum(decode_costs[:warmup]) +
            np.mean(decode_costs[warmup:]) * (gen_len - 1 - warmup))

    return decode_latency


def write_benchmark_log(filename, model_size, cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput):
    """
    将基准测试结果写入日志文件。
    参数：
        filename: 日志文件路径
        model_size: 模型权重总大小（字节）
        cache_size: 缓存总大小（字节）
        hidden_size: 隐藏状态总大小（字节）
        gpu_peak_mem: GPU峰值内存占用（字节）
        projected: 是否为预测结果（布尔值）
        其余参数：各阶段延迟和吞吐量
    返回：
        写入的日志字符串
    """
    log_str = (f"model size: {model_size/GB:.3f} GB\t"  # 模型大小（GB）
               f"cache size: {cache_size/GB:.3f} GB\t"  # 缓存大小（GB）
               f"hidden size (p): {hidden_size/GB:.3f} GB\n"  # 隐藏状态大小（GB）
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t"  # GPU峰值内存（GB）
               f"projected: {projected}\n"  # 是否为预测结果
               f"prefill latency: {prefill_latency:.3f} s\t"  # 预填充延迟
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"  # 预填充吞吐量
               f"decode latency: {decode_latency:.3f} s\t"  # 解码延迟
               f"decode throughput: {decode_throughput:.3f} token/s\n"  # 解码吞吐量
               f"total latency: {total_latency:.3f} s\t"  # 总延迟
               f"total throughput: {total_throughput:.3f} token/s")  # 总吞吐量
    # 追加写入日志文件
    with open(filename, "a") as fout:
        fout.write(log_str + "\n")

    return log_str


def read_benchmark_log(filename):
    """
    从日志文件读取基准测试结果并解析为BenchmarkResult对象。
    参数：
        filename: 日志文件路径
    返回：
        BenchmarkResult实例（包含各阶段延迟和吞吐量）
    """
    with open(filename) as fin:
        lines = fin.readlines()

    def extract(line):
        """从日志行中提取延迟和吞吐量。"""
        a, b = line.split("\t")
        latency = a[a.index(":") + 1:a.index(" s")]  # 提取延迟值（排除" s"）
        throughput = b[b.index(":") + 1:b.index(" to")]  # 提取吞吐量值（排除" token/s"）
        return float(latency), float(throughput)

    # 解析日志行（第2行为预填充，第3行为解码，第4行为总指标）
    prefill_latency, prefill_throughput = extract(lines[2])
    decode_latency, decode_throughput = extract(lines[3])
    total_latency, total_throughput = extract(lines[4])

    return BenchmarkResult(
        prefill_latency, prefill_throughput,
        decode_latency, decode_throughput,
        total_latency, total_throughput,
    )