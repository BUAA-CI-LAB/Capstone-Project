"""
4位组量化压缩设备实现
用于管理张量的压缩存储（量化）和解压缩（反量化），支持CPU/GPU设备，
核心应用于注意力缓存（K/V Cache）的存储优化，降低显存/内存占用
"""

import dataclasses

import torch
import numpy as np

# 导入自定义张量、设备相关模块
from flexllmgen.pytorch_backend import (TorchTensor, TorchDevice,
    DeviceType, general_copy, fix_recursive_import)
from flexllmgen.utils import np_dtype_to_torch_dtype


@dataclasses.dataclass
class CompressionConfig:
    """
    组量化配置类：定义4位组量化的核心参数
    仅支持非对称4位量化（当前实现限制）
    """
    num_bits: int  # 量化位数（当前仅支持4位）
    group_size: int  # 分组大小（每个组包含的元素数，需为偶数）
    group_dim: int  # 分组维度（沿哪个维度进行分组量化）
    symmetric: bool  # 是否对称量化（当前仅支持False，非对称量化）
    enabled: bool = True  # 是否启用压缩（默认启用）


class TorchCompressedDevice:
    """
    压缩设备类：基于基础设备（CPU/GPU）实现张量的压缩存储与解压缩计算
    核心功能：4位非对称组量化，将float16张量压缩为uint8（数据）+ float16（缩放因子）
    """

    def __init__(self, base_device):
        """
        初始化压缩设备
        Args:
            base_device: 基础物理设备（TorchDevice实例，如CPU/GPU）
        """
        self.name = "compressed"  # 设备名称
        self.device_type = DeviceType.COMPRESSED  # 设备类型为压缩设备
        self.base_device = base_device  # 底层物理设备

        self.data_decompress_workspace = None  # 解压缩工作空间（CPU专用）
        self.workspace_pt = 0  # 工作空间轮询指针（循环使用避免重复分配）

    def allocate(self, shape, dtype, comp_config, pin_memory=None, name=None):
        """
        分配压缩格式的TorchTensor（预分配数据和缩放因子存储）
        形状会向上对齐到分组大小的整数倍，确保量化时无需动态padding
        Args:
            shape: 原始张量形状（压缩前）
            dtype: 原始数据类型（仅支持np.float16）
            comp_config: 量化配置（CompressionConfig实例）
            pin_memory: 是否使用锁页内存（缓存场景禁用，避免内存开销）
            name: 张量名称（可选）
        Returns:
            压缩格式的TorchTensor（data=(压缩数据, 缩放因子, 配置)）
        """
        # 仅支持4位量化和float16原始类型（当前实现限制）
        assert comp_config.num_bits == 4 and dtype == np.float16

        group_size, group_dim = comp_config.group_size, comp_config.group_dim

        # 计算压缩后的数据形状：4位量化→每个uint8存储2个元素，故维度长度=组数×(group_size//2)
        num_groups = (shape[group_dim] + group_size - 1) // group_size  # 向上取整计算组数
        data_shape = (
            shape[:group_dim] + (num_groups * (group_size // 2),) + shape[group_dim+1:])
        # 计算缩放因子形状：每组存储1个scale和1个min（共2个float16值）
        scale_shape = (
            shape[:group_dim] + (num_groups, 2) + shape[group_dim+1:])

        # 在基础设备上分配存储：数据用uint8（压缩），缩放因子用float16
        data = self.base_device.allocate(data_shape, np.uint8, pin_memory=pin_memory)
        scale = self.base_device.allocate(scale_shape, np.float16, pin_memory=pin_memory)

        # 返回压缩张量：存储数据、缩放因子和量化配置
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (data, scale, comp_config), self, name=name)

    def init_cache_one_gpu_batch(self, config, task, policy):
        """
        初始化单GPU批处理的压缩K/V缓存
        Args:
            config: 模型配置（n_head, input_dim等）
            task: 推理任务配置（prompt_len, gen_len等）
            policy: 推理策略（包含comp_cache_config量化配置）
        Returns:
            (k_cache, v_cache)：压缩格式的K/V缓存张量
        """
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        # 缓存形状：(最大序列长度, 批大小×头数, 头维度)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # 禁用锁页内存：缓存数据量大，锁页内存开销过高
        pin_memory = False
        # 分配压缩缓存（使用策略中的量化配置）
        k_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        return k_cache, v_cache

    def init_attention_compute_workspace(self, config, task, policy):
        """
        初始化注意力计算的解压缩工作空间（仅CPU需要）
        CPU设备需预分配fp32工作空间，避免解压缩时重复分配内存
        Args:
            config: 模型配置
            task: 推理任务配置
            policy: 推理策略
        """
        if self.base_device.device_type != DeviceType.CPU:
            return  # 仅CPU需要fp32工作空间（GPU可动态分配）

        # 计算工作空间形状（基于最大序列长度和模型参数）
        b = policy.gpu_batch_size
        n_head = config.n_head
        head_dim = config.input_dim // n_head
        max_seq_len = task.prompt_len + task.gen_len - 1
        shape = (max_seq_len, b * n_head, head_dim)

        # 按分组维度调整形状（对齐到组大小）
        group_size, group_dim = (
            policy.comp_cache_config.group_size, policy.comp_cache_config.group_dim)
        num_groups = (shape[group_dim] + group_size - 1) // group_size
        new_shape = (shape[:group_dim] + (num_groups, group_size) +
                     shape[group_dim+1:])

        # 分配两个循环使用的工作空间（避免频繁申请释放）
        self.data_decompress_workspace = [
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
        ]

    def compress(self, tensor, comp_config):
        """
        将PyTorch张量压缩为4位组量化格式
        核心步骤：Pad→分组→计算scale/min→量化→打包（2个4位值→1个uint8）
        Args:
            tensor: 输入张量（torch.Tensor，float16）
            comp_config: 量化配置（CompressionConfig实例）
        Returns:
            压缩格式的TorchTensor（data=(压缩数据, 缩放因子, 配置)）
        """
        group_size, num_bits, group_dim, symmetric = (
            comp_config.group_size, comp_config.num_bits,
            comp_config.group_dim, comp_config.symmetric)
        # 校验：仅支持4位、偶数组大小、非对称量化
        assert num_bits == 4 and group_size % 2 == 0 and not symmetric

        # CPU上的float16张量需转为float32计算（避免精度损失）
        if tensor.device.type == "cpu" and tensor.dtype == torch.float16:
            tensor = tensor.float()

        shape = tensor.shape
        num_groups = (shape[group_dim] + group_size - 1) // group_size  # 向上取整计算组数

        # Step 1: Pad到组大小的整数倍（确保所有元素都能分组）
        new_shape = (shape[:group_dim] + (num_groups, group_size) +
                     shape[group_dim+1:])
        pad_len = (group_size - shape[group_dim] % group_size) % group_size
        if pad_len != 0:
            pad_shape = shape[:group_dim] + (pad_len,) + shape[group_dim+1:]
            tensor = torch.cat([
                tensor,
                torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
                dim=group_dim)
        data = tensor.view(new_shape)  # 重塑为(..., 组数, 组大小, ...)

        # Step 2: 非对称量化计算
        B = 2 ** num_bits - 1  # 4位量化的最大值（0~15）
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]  # 每组最小值
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]  # 每组最大值
        scale = B / (mx - mn)  # 缩放因子（将组内值映射到0~15）

        # 量化：减去最小值→乘以缩放因子→裁剪到0~B→四舍五入→转为uint8
        data = data - mn
        data.mul_(scale)
        data = data.clamp_(0, B).round_().to(torch.uint8)

        # Step 3: 打包（2个4位值→1个uint8）
        # 取每组中偶数索引元素（左4位）和奇数索引元素（右4位）
        left_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(0, data.shape[group_dim+1], 2),))  # 步长2取左元素
        right_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(1, data.shape[group_dim+1], 2),))  # 步长2取右元素
        # 左元素左移4位 + 右元素（与0xF掩码）→ 打包为uint8
        data = torch.bitwise_or(
            data[left_indices].bitwise_left_shift(4), data[right_indices])

        # Step 4: 重塑为最终压缩形状
        data_shape = (
            shape[:group_dim] + (num_groups * (group_size // 2),) + shape[group_dim+1:])
        scale_shape = (
            shape[:group_dim] + (num_groups, 2) + shape[group_dim+1:])
        data = data.view(data_shape)  # 压缩数据形状
        scale = torch.cat([scale, mn], dim=group_dim+1).view(scale_shape)  # 合并scale和mn

        # 转换为TorchTensor格式
        data = TorchTensor.create_from_torch(data, self.base_device)
        scale = TorchTensor.create_from_torch(scale, self.base_device)

        # 返回压缩张量（包含原始形状和 dtype）
        return TorchTensor(shape, tensor.dtype,
                           (data, scale, comp_config), self)

    def decompress(self, tensor):
        """
        将压缩格式的TorchTensor解压缩为原始float16张量
        核心步骤：解包→反量化→Unpad→重塑为原始形状
        Args:
            tensor: 压缩格式的TorchTensor
        Returns:
            解压缩后的PyTorch张量（float16，原始形状）
        """
        # 解析压缩张量的组成部分
        data, scale, comp_config = tensor.data
        group_size, num_bits, group_dim, symmetric = (
            comp_config.group_size, comp_config.num_bits,
            comp_config.group_dim, comp_config.symmetric)

        group_size_c = group_size // 2  # 压缩后每组的uint8数量（2个4位值→1个uint8）
        shape = data.shape
        num_groups = (shape[group_dim] + group_size_c - 1) // group_size_c  # 组数

        # Step 1: Pad压缩数据到组大小的整数倍（确保解包时无残留）
        new_shape = (shape[:group_dim] + (num_groups, group_size_c) +
                     shape[group_dim+1:])
        pad_len = (group_size_c - shape[group_dim] % group_size_c) % group_size_c
        if pad_len != 0:
            pad_shape = shape[:group_dim] + (pad_len,) + shape[group_dim+1:]
            data = torch.cat([
                data,
                torch.zeros(pad_shape, dtype=data.dtype, device=data.device)],
                dim=group_dim)
        packed = data.data.view(new_shape)  # 重塑为(..., 组数, 压缩组大小, ...)

        # Step 2: 解包（1个uint8→2个4位值）
        if self.base_device.device_type == DeviceType.CPU:
            # CPU使用预分配的工作空间（循环使用）
            self.workspace_pt = (self.workspace_pt + 1) % len(
                self.data_decompress_workspace)
            data = self.data_decompress_workspace[self.workspace_pt][:shape[0]]
        else:
            # GPU动态分配解包空间（float16）
            new_shape = (shape[:group_dim] + (num_groups, group_size,) +
                         shape[group_dim+1:])
            data = torch.empty(new_shape, dtype=torch.float16, device=packed.device)
        
        # 左4位：右移4位；右4位：与0xF掩码（提取低4位）
        left_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(0, data.shape[group_dim+1], 2),))  # 偶数索引存左4位
        right_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(1, data.shape[group_dim+1], 2),))  # 奇数索引存右4位
        data[left_indices] = packed.bitwise_right_shift(4)
        data[right_indices] = packed.bitwise_and(0xF)

        # Step 3: 反量化（恢复为原始值范围）
        scale, mn = scale.data.split(1, dim=group_dim + 1)  # 拆分scale和mn
        data.div_(scale)  # 除以缩放因子
        data.add_(mn)     # 加回最小值

        # Step 4: Unpad（移除压缩时添加的padding）
        unpad_len = (group_size - tensor.shape[group_dim] % group_size) % group_size
        if unpad_len != 0:
            flatten_shape = (shape[:group_dim] + (num_groups * group_size,) +
                             shape[group_dim+1:])
            indices = [slice(0, x) for x in flatten_shape]
            indices[group_dim] = slice(0, flatten_shape[group_dim] - unpad_len)
            data = data.view(flatten_shape)[indices].contiguous()

        # 重塑为原始形状并返回
        return data.view(tensor.shape)


def general_copy_compressed(dst, dst_indices, src, src_indices):
    """
    压缩张量的通用拷贝函数：分别拷贝压缩数据和缩放因子
    需计算压缩后数据和缩放因子对应的索引（因分组和打包导致索引映射变化）
    Args:
        dst: 目标压缩张量（TorchTensor）
        dst_indices: 目标张量索引（slice元组）
        src: 源压缩张量（TorchTensor）
        src_indices: 源张量索引（slice元组）
    """
    # 校验：源和目标必须都是压缩设备张量
    assert (src.device.device_type == DeviceType.COMPRESSED and
            dst.device.device_type == DeviceType.COMPRESSED)

    # 计算源张量的压缩数据和缩放因子索引
    src_data_indices, src_scale_indices = get_compressed_indices(
        src, src_indices, src.shape)

    # 计算目标张量的压缩数据和缩放因子索引
    dst_data_indices, dst_scale_indices = get_compressed_indices(
        dst, dst_indices, dst.shape)

    # 分别拷贝压缩数据和缩放因子（调用通用拷贝函数）
    general_copy(dst.data[0], dst_data_indices, src.data[0], src_data_indices)
    general_copy(dst.data[1], dst_scale_indices, src.data[1], src_scale_indices)


def get_compressed_indices(tensor, indices, shape):
    """
    计算压缩张量中数据和缩放因子对应的索引（基于原始索引）
    核心逻辑：根据量化配置调整索引（分组维度的索引映射）
    Args:
        tensor: 压缩张量（TorchTensor）
        indices: 原始索引（slice元组，针对压缩前形状）
        shape: 压缩前的原始形状
    Returns:
        (data_indices, scale_indices)：压缩数据和缩放因子的索引
    """
    comp_config = tensor.data[2]
    group_size, group_dim = comp_config.group_size, comp_config.group_dim
    assert comp_config.num_bits == 4  # 仅支持4位量化

    # 补全默认索引（若未指定则取整个维度）
    if indices is None:
        indices = list(slice(0, x) for x in shape[:group_dim+1])
    else:
        indices = list(indices) + [slice(0, x) for x in shape[len(indices):]]
    # 校验：原始索引的分组维度起始位置必须是组大小的整数倍（确保分组对齐）
    assert indices[group_dim].start % group_size == 0

    # 计算压缩数据的索引：4位打包→分组维度长度减半
    data_indices = list(indices)
    data_indices[group_dim] = slice(
        indices[group_dim].start // 2,  # 起始位置//2
        (indices[group_dim].stop + 1) // 2  # 结束位置向上取整//2
    )

    # 计算缩放因子的索引：每组对应1个scale+mn，故分组维度长度=组数
    scale_indices = indices
    scale_indices.insert(group_dim+1, slice(0, 2))  # 插入scale/mn维度（长度2）
    scale_indices[group_dim] = slice(
        indices[group_dim].start // group_size,  # 起始组号
        (indices[group_dim].stop + group_size - 1) // group_size  # 结束组号（向上取整）
    )

    return data_indices, scale_indices


# 默认缓存压缩配置（禁用压缩）
default_cache_config = CompressionConfig(
    num_bits=0, group_size=0, group_dim=0, symmetric=False, enabled=False)


def set_cache_compression_config(config):
    """设置全局默认的缓存压缩配置"""
    global default_cache_config
    default_cache_config = config


def get_cache_compression_config():
    """获取全局默认的缓存压缩配置"""
    return default_cache_config


def compress(tensor, config):
    """
    模拟组量化压缩（独立于设备的纯函数版本）
    用于测试或简单场景，不涉及TorchTensor和设备管理
    Args:
        tensor: 输入张量（torch.Tensor）
        config: 量化配置（CompressionConfig实例）
    Returns:
        压缩后的数据（量化值+缩放参数+原始形状）
    """
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)
    assert num_bits <= 8  # 支持最多8位量化

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (original_shape[:group_dim] + (num_groups, group_size) +
                 original_shape[group_dim+1:])

    # Step 1: Pad到组大小的整数倍
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = original_shape[:group_dim] + (pad_len,) + original_shape[group_dim+1:]
        tensor = torch.cat([
            tensor,
            torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim)
    data = tensor.view(new_shape)

    # Step 2: 量化计算
    if symmetric:
        # 对称量化（适用于权重，取值范围对称）
        B = 2 ** (num_bits - 1) - 1  # 对称量化的最大值（如4位：-7~7）
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        # 非对称量化（适用于激活值/缓存，取值范围0~B）
        B = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    """
    模拟组量化反压缩（对应compress函数的逆操作）
    Args:
        packed_data: 压缩数据（compress函数的输出）
        config: 量化配置（CompressionConfig实例）
    Returns:
        解压缩后的张量（原始形状和数据类型）
    """
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)

    # Step 1: 反量化
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)

    # Step 2: Unpad（移除压缩时的padding）
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim] +
            (original_shape[group_dim] + pad_len,) +
            original_shape[group_dim+1:])
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)


def compress_and_decompress(tensor, config):
    """压缩-解压缩闭环测试函数：验证量化的可逆性"""
    packed_data = compress(tensor, config)
    return decompress(packed_data, config)


def test_simulated_compression():
    """测试模拟量化（compress/decompress函数）"""
    torch.manual_seed(0)  # 固定随机种子，确保结果可复现
    # 生成测试张量（float16，GPU上）
    a = torch.normal(0, 1, (64, 64, 64), dtype=torch.float16).cuda()

    # 量化配置：4位、32组大小、第0维分组、非对称量化
    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    packed_data = compress(a, config)
    b = decompress(packed_data, config)

    # 打印前几个元素，验证量化前后的一致性（允许微小误差）
    print("模拟量化前张量前10个元素：")
    print(a[0, 0, :10])
    print("模拟量化后解压缩张量前10个元素：")
    print(b[0, 0, :10])


def test_real_compression():
    """测试真实设备量化（TorchCompressedDevice的compress/decompress）"""
    torch.manual_seed(0)
    # 生成测试张量（float16，GPU上）
    a = torch.normal(0, 1, (32, 1, 1), dtype=torch.float16).cuda()

    # 量化配置：4位、32组大小、第0维分组、非对称量化
    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    # 创建GPU压缩设备
    dev = TorchDevice("cuda:0", 0, 0).compressed_device
    # 压缩→解压缩
    packed = dev.compress(a, config)
    b = dev.decompress(packed)

    # 打印结果，验证一致性
    print("真实设备量化前张量：")
    print(a.flatten())
    print("真实设备量化后解压缩张量：")
    print(b.flatten())


if __name__ == "__main__":
    fix_recursive_import()  # 修复循环导入问题
    # test_simulated_compression()  # 测试模拟量化
    test_real_compression()  # 测试真实设备量化