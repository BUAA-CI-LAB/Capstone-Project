"""使用PyTorch实现张量计算（FlexGen框架核心后端）"""
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# 导入FlexGen工具函数：内存单位转换、张量操作、数据类型转换等
from flexllmgen.utils import (
    GB, T,  # 内存单位（GB）和张量标记
    cpu_mem_stats,  # CPU内存状态查询
    vector_gather,  # 向量收集操作
    np_dtype_to_torch_dtype,  # numpy数据类型转PyTorch
    torch_dtype_to_np_dtype,  # PyTorch数据类型转numpy
    torch_dtype_to_num_bytes  # PyTorch数据类型占字节数
)

# 全局变量：压缩相关函数/类、全局CPU/磁盘设备（延迟初始化）
general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    """修复循环导入问题：延迟导入压缩相关模块（避免初始化顺序冲突）"""
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexllmgen import compression
    general_copy_compressed = compression.general_copy_compressed  # 通用压缩拷贝函数
    TorchCompressedDevice = compression.TorchCompressedDevice  # 压缩设备类


class DeviceType(Enum):
    """设备类型枚举：定义FlexGen支持的所有设备类型"""
    CPU = auto()  # CPU设备
    CUDA = auto()  # GPU设备（CUDA）
    DISK = auto()  # 磁盘设备（用于Offloading存储）
    MIXED = auto()  # 混合设备（部分数据在GPU、部分在CPU/磁盘）
    COMPRESSED = auto()  # 压缩设备（张量压缩存储）

    @staticmethod
    def convert(name):
        """将设备名称字符串转换为DeviceType枚举值"""
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"无效的设备名称: {name}")


class TorchTensor:
    """
    统一张量封装类：封装不同设备（GPU/CPU/磁盘/混合/压缩）的张量操作
    核心作用：让上层代码无需关心数据存储位置，提供统一的张量接口（拷贝、迁移、加载等）
    
    不同设备的data存储格式：
    - CPU/GPU（TorchDevice）：data是PyTorch张量（torch.Tensor）
    - 磁盘（TorchDisk）：data是文件路径（字符串）
    - 混合设备（TorchMixedDevice）：data是(子张量元组, 分段点元组)
    - 压缩设备（TorchCompressedDevice）：data是(数据张量, 缩放张量, 压缩配置)
    """
    name_count = count()  # 张量名称计数器（自动生成唯一名称）

    def __init__(self, shape, dtype, data, device, name=None):
        """
        初始化张量
        Args:
            shape: 张量形状（如(16384, 4096)）
            dtype: 数据类型（如torch.float16）
            data: 原始数据（根据设备类型不同，格式不同）
            device: 张量所属设备（TorchDevice/TorchDisk等）
            name: 张量名称（可选，默认自动生成）
        """
        # 校验：若data是PyTorch张量，需确保其设备与当前张量设备一致
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape  # 张量形状
        self.dtype = dtype  # 数据类型
        self.data = data    # 原始数据（格式随设备变化）
        self.device = device  # 所属设备

        # 张量删除时是否自动删除磁盘文件（仅磁盘设备有效）
        self.delete_file = True

        # 张量名称（默认生成格式：t_0, t_1, ...）
        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        """计算张量占用的字节数（形状乘积 × 单个数据类型字节数）"""
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        """生成下一个唯一张量名称"""
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        """从PyTorch张量创建TorchTensor实例"""
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        """删除张量（释放设备资源）"""
        assert self.device is not None, "张量已被删除"
        # 若为磁盘设备，调用设备的delete方法删除文件
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        # 置空设备和数据引用（触发垃圾回收）
        self.device = self.data = None

    def load_from_np(self, np_array):
        """从numpy数组加载数据到张量"""
        if self.device.device_type == DeviceType.DISK:
            # 磁盘设备：将numpy数组保存为文件
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                # 压缩设备：先将numpy数组转为PyTorch张量，再压缩
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                # CPU/GPU设备：直接拷贝numpy数组到张量
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        """从numpy文件（.npy）加载数据到张量"""
        if self.device.device_type == DeviceType.DISK:
            # 磁盘设备：直接拷贝文件到目标路径
            shutil.copy(filename, self.data)
        else:
            # 其他设备：先加载numpy文件，再调用load_from_np
            self.load_from_np(np.load(filename))

    def copy(self, dst, src_indices=None):
        """
        将当前张量拷贝到目标设备（dst）
        Args:
            dst: 目标设备（TorchDevice/TorchDisk等）
            src_indices: 源张量切片索引（可选，如[slice(0,100), slice(0,200)]）
        Returns:
            目标设备上的新张量
        """
        # 计算目标张量形状（若指定切片，按切片计算形状）
        if src_indices:
            # 校验：切片必须是连续的（无步长）
            assert all(x.step is None for x in src_indices)
            # 切片部分形状 + 剩余维度形状
            shape = tuple(x.stop - x.start for x in src_indices) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        # 为目标设备分配张量内存
        if dst.device_type == DeviceType.COMPRESSED:
            # 压缩设备：需传入压缩配置
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            # 其他设备：直接分配内存
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        # 执行拷贝操作（通用拷贝函数，支持跨设备）
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        """
        智能拷贝：若当前张量已在目标设备，直接返回自身；否则执行拷贝
        Args:
            dst: 目标设备
            src_indices: 源张量切片索引（可选）
        Returns:
            (目标张量, 是否执行了拷贝)
        """
        if self.device == dst:
            return self, False  # 已在目标设备，无需拷贝
        return self.copy(dst, src_indices=src_indices), True  # 执行拷贝

    def move(self, dst):
        """
        将张量迁移到目标设备（原设备上的张量会被删除）
        Args:
            dst: 目标设备
        Returns:
            目标设备上的新张量
        """
        if self.device == dst:
            return self  # 已在目标设备，无需迁移
        # 拷贝到目标设备，删除原张量
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        """张量字符串表示（便于调试）"""
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")


class TorchDevice:
    """单设备管理类：封装单个CPU/GPU的张量操作和计算接口"""

    def __init__(self, name, mem_capacity=None, flops=None):
        """
        初始化设备
        Args:
            name: 设备名称（如"cpu", "cuda:0"）
            mem_capacity: 设备内存容量（如8GB）
            flops: 设备计算能力（浮点运算次数/秒，用于性能估算）
        """
        self.name = name  # 设备名称
        self.mem_capacity = mem_capacity  # 内存容量
        self.flops = flops  # 计算能力

        self.dev = torch.device(name)  # PyTorch设备对象
        self.device_type = DeviceType.convert(self.dev.type)  # 设备类型（DeviceType枚举）
        self.compressed_device = TorchCompressedDevice(self)  # 关联的压缩设备

        self.links = {}  # 设备间连接（用于跨设备数据传输优化）

        self.attention_compute_workspace = None  # 注意力计算工作空间（CPU专用）
        self.workspace_pt = 0  # 工作空间指针（循环使用多个工作空间）

        # 若为CPU设备，设置为全局CPU设备（供其他模块调用）
        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        """添加设备间连接（用于优化跨设备数据传输）"""
        # 确定连接的目标设备
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        """
        在设备上分配张量内存
        Args:
            shape: 张量形状
            dtype: 数据类型（numpy格式）
            pin_memory: 是否启用锁页内存（仅CPU有效，加速CPU→GPU传输）
            name: 张量名称（可选）
        Returns:
            分配的TorchTensor实例
        """
        # 锁页内存配置：CPU默认启用，其他设备禁用
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        # 转换数据类型（numpy→PyTorch）
        dtype = np_dtype_to_torch_dtype[dtype]
        # 分配空张量（未初始化数据）
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        # 创建TorchTensor实例并返回
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        """删除设备上的张量（空实现，PyTorch会自动垃圾回收）"""
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        """
        初始化注意力计算工作空间（仅CPU需要，用于存储k/v缓存的浮点32中间结果）
        Args:
            config: 模型配置（如n_head, input_dim）
            task: 推理任务配置（如prompt_len, gen_len）
            policy: 推理策略（如gpu_batch_size, compress_cache）
        """
        # 仅CPU需要初始化工作空间（GPU无需）
        if self.device_type != DeviceType.CPU:
            return

        if not policy.compress_cache:
            # 非压缩缓存：计算工作空间形状
            b = policy.gpu_batch_size  # 单GPU批大小
            n_head = config.n_head  # 注意力头数
            head_dim = config.input_dim // n_head  # 每个头的维度
            max_seq_len = task.prompt_len + task.gen_len - 1  # 最大序列长度（输入+生成）
            self.attention_compute_workspace = []  # 工作空间列表（存储k/v缓存对）
            self.workspace_pt = 0  # 工作空间指针初始化

            # 工作空间数量：若分离注意力层和MLP层，需2个；否则1个
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)  # 工作空间形状
                k_cache = self.allocate(shape, np.float32, pin_memory=False)  # k缓存
                v_cache = self.allocate(shape, np.float32, pin_memory=False)  # v缓存
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            # 压缩缓存：调用压缩设备初始化工作空间
            self.compressed_device.init_attention_compute_workspace(config, task, policy)

    def next_attention_compute_workspace(self):
        """循环获取下一个注意力计算工作空间（避免重复分配）"""
        self.workspace_pt = (self.workspace_pt + 1) % len(self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        """删除注意力计算工作空间（释放内存）"""
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        """
        生成注意力掩码（屏蔽填充token）
        Args:
            token_ids: 输入token ID张量
            pad_token_id: 填充token的ID（如0）
            donate: 是否释放输入张量内存（[True/False]）
        Returns:
            注意力掩码张量（True表示有效token，False表示填充token）
        """
        # 生成掩码：token_id != pad_token_id 的位置为True
        data = token_ids.data.ne(pad_token_id)
        # 若允许释放，删除输入token_ids张量
        if donate[0]:
            token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        """
        扩展注意力掩码（生成阶段新增token时，在掩码末尾添加True）
        Args:
            attention_mask: 原始注意力掩码
            donate: 是否释放原始掩码内存（[True/False]）
        Returns:
            扩展后的注意力掩码
        """
        bs = attention_mask.shape[0]  # 批大小
        # 在掩码最后一列添加全1（新生成的token是有效token）
        data = torch.concat((
            attention_mask.data,
            torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)
        ), dim=1)
        # 若允许释放，删除原始掩码
        if donate[0]:
            attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        """
        OPT模型输入嵌入层：将token ID转换为模型输入张量（token嵌入 + 位置嵌入）
        Args:
            inputs: 输入token ID张量（shape: (batch_size, seq_len)）
            attention_mask: 注意力掩码（shape: (batch_size, seq_len)）
            w_token: token嵌入权重（shape: (vocab_size, hidden_dim)）
            w_pos: 位置嵌入权重（shape: (max_seq_len, hidden_dim)）
            pad_token_id: 填充token ID
            donate: 是否释放输入张量内存（[释放inputs, 释放attention_mask]）
        Returns:
            嵌入后的输入张量（shape: (batch_size, seq_len, hidden_dim)）
        """
        # 若权重是压缩的，先解压缩
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data  # token ID原始数据
        mask = attention_mask.data  # 掩码原始数据
        # 释放输入张量内存（若允许）
        if donate[0]:
            inputs.delete()
        if donate[1]:
            attention_mask.delete()

        # 1. Token嵌入：通过嵌入层将token ID映射为向量
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        # 2. 位置嵌入：生成每个token的位置索引，再通过嵌入层映射
        # 计算位置：掩码累积和（有效token位置从1开始计数）
        positions = torch.cumsum(mask, dim=1).int() * mask + 1
        # 若存在历史key/value缓存，裁剪位置索引（从缓存长度后开始计数）
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]
        # 位置嵌入映射
        pos_embed = F.embedding(positions, w_pos.data)

        # 3. 合并token嵌入和位置嵌入
        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate, do_sample, temperature):
        """
        OPT模型输出嵌入层：将模型隐藏态转换为token ID（含归一化、线性映射、采样）
        Args:
            inputs: 模型最后一层隐藏态（shape: (batch_size, seq_len, hidden_dim)）
            w_ln: 层归一化权重（shape: (hidden_dim,)）
            b_ln: 层归一化偏置（shape: (hidden_dim,)）
            w_token: 输出投影权重（与token嵌入权重共享，shape: (vocab_size, hidden_dim)）
            donate: 是否释放输入张量内存（[True/False]）
            do_sample: 是否启用采样生成（True=采样，False=贪心搜索）
            temperature: 采样温度（控制随机性，越低越确定）
        Returns:
            生成的token ID张量（shape: (batch_size, 1)）
        """
        # 若权重是压缩的，先解压缩
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape  # batch_size, seq_len, hidden_dim

        # 1. 层归一化：对隐藏态进行归一化
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        # 释放输入张量内存（若允许）
        if donate[0]:
            inputs.delete()

        # 2. 输出投影：将隐藏态映射到词汇表维度（logits）
        logits = F.linear(hidden, w_token.data)
        # 取最后一个token的logits（生成下一个token）
        last_token_logits = logits[:, -1, :]

        # 3. 生成token ID：采样或贪心搜索
        if do_sample and not temperature < 1e-5:
            # 采样生成：softmax转换为概率，再按概率采样
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            # 贪心搜索：取概率最大的token
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy):
        """
        初始化单GPU批处理的注意力缓存（k/v缓存）
        Args:
            config: 模型配置（n_head, input_dim）
            task: 推理任务配置（prompt_len, gen_len）
            policy: 推理策略（gpu_batch_size）
        Returns:
            (k_cache, v_cache)：k缓存和v缓存张量
        """
        # 解构配置参数
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len, policy.gpu_batch_size
        )
        # 缓存形状：(最大序列长度, 批大小×头数, 头维度)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # 禁用锁页内存（减少内存开销）
        pin_memory = False
        # 分配k/v缓存（float16类型，节省显存）
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache

    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """
        多头注意力机制（预处理阶段，Prefill）：处理输入序列的所有token
        Args:
            inputs: 嵌入后的输入张量（shape: (batch_size, seq_len, hidden_dim)）
            attention_mask: 注意力掩码（shape: (batch_size, seq_len)）
            w_q/w_k/w_v: 查询/键/值投影权重（shape: (hidden_dim, hidden_dim)）
            b_q/b_k/b_v: 查询/键/值投影偏置（shape: (hidden_dim,)）
            w_out: 输出投影权重（shape: (hidden_dim, hidden_dim)）
            b_out: 输出投影偏置（shape: (hidden_dim,)）
            w_ln/b_ln: 层归一化权重/偏置（shape: (hidden_dim,)）
            n_head: 注意力头数
            donate: 是否释放输入张量内存（[释放inputs, 释放attention_mask]）
            compress_cache: 是否压缩缓存
            comp_config: 压缩配置
        Returns:
            (output, k_cache, v_cache)：注意力输出、k缓存、v缓存
        """
        # 若权重是压缩的，先解压缩
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape  # batch_size, seq_len, hidden_dim
        head_dim = h // n_head  # 每个注意力头的维度
        scaling = head_dim ** -0.5  # 缩放因子（避免logits过大）

        # 1. 层归一化
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # 2. 投影：将隐藏态转换为查询（q）、键（k）、值（v）
        # shape: (batch_size, seq_len, hidden_dim)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)

        # 3. 多头拆分：将hidden_dim拆分为n_head个head_dim
        # shape: (batch_size, seq_len, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # 4. 维度重排：适配批量矩阵乘法（bmm）
        # q: (batch_size × n_head, seq_len, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # k: (batch_size × n_head, head_dim, seq_len)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # v: (batch_size × n_head, seq_len, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # 5. 计算注意力权重：q × k^T
        # shape: (batch_size × n_head, seq_len, seq_len)
        attn_weights = torch.bmm(q, k)

        # 6. 掩码处理：屏蔽填充token和未来token（因果掩码）
        # 因果掩码：上三角矩阵（避免当前token关注未来token）
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        # 合并填充掩码和因果掩码
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # 7. 注意力权重归一化：softmax
        attn_weights = attn_weights.view(b, n_head, s, s)  # 恢复head维度
        attn_weights = torch.where(mask, attn_weights, -1e4)  # 掩码位置设为极小值
        attn_weights = attn_weights.view(b * n_head, s, s)  # 重排为批量格式
        attn_weights = F.softmax(attn_weights, dim=2)  # 按最后一维归一化

        # 8. 计算注意力输出：attn_weights × v
        # shape: (batch_size × n_head, seq_len, head_dim)
        value = torch.bmm(attn_weights, v)
        # 维度重排：恢复原始形状
        value = value.view(b, n_head, s, head_dim)  # (batch_size, n_head, seq_len, head_dim)
        value = value.transpose(1, 2).reshape(b, s, h)  # (batch_size, seq_len, hidden_dim)

        # 9. 输出投影：将多头输出映射回hidden_dim
        value = F.linear(value, w_out.data, bias=b_out.data)

        # 10. 残差连接：加上原始输入（归一化前）
        value.add_(inputs.data)

        # 释放输入张量内存（若允许）
        if donate[0]:
            inputs.delete()
        if donate[1]:
            attention_mask.delete()

        # 11. 准备k/v缓存：维度重排为（seq_len, batch_size×n_head, head_dim）
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        # 若启用缓存压缩，对k/v缓存进行压缩
        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            # 封装为TorchTensor实例
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        # 返回注意力输出、k缓存、v缓存
        return TorchTensor.create_from_torch(value, self), k, v

    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """
        多头注意力机制（生成阶段，Decoding）：逐token生成，复用历史k/v缓存
        Args:
            inputs: 上一轮生成的token嵌入（shape: (batch_size, 1, hidden_dim)）
            attention_mask: 注意力掩码（shape: (batch_size, current_seq_len)）
            w_q/w_k/w_v/w_out: 投影权重（同mha方法）
            b_q/b_k/b_v/b_out: 投影偏置（同mha方法）
            w_ln/b_ln: 层归一化权重/偏置（同mha方法）
            n_head: 注意力头数
            k_cache/v_cache: 历史k/v缓存（存储已生成token的键/值）
            donate: 是否释放输入张量内存（[释放inputs, 释放attention_mask]）
            attn_sparsity: 注意力稀疏度（1.0=稠密注意力，<1.0=稀疏注意力）
            compress_cache: 是否压缩缓存
            comp_config: 压缩配置
        Returns:
            (output, new_k_cache, new_v_cache)：注意力输出、更新后的k/v缓存
        """
        # 若权重是压缩的，先解压缩
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape  # batch_size, 1（单token）, hidden_dim
        src_s = attention_mask.shape[1]  # 当前序列长度（输入+已生成token）
        head_dim = h // n_head  # 每个头的维度
        scaling = head_dim ** -0.5  # 缩放因子

        # 1. 层归一化
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # 2. 投影：生成当前token的q、k、v
        # shape: (batch_size, 1, hidden_dim)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)

        # 3. 多头拆分与维度重排
        # shape: (batch_size, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # q: (batch_size × n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # 新生成的k/v：(1, batch_size × n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        # 4. 处理k/v缓存：更新历史缓存并计算注意力
        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # 稠密注意力（关注所有历史token）
                if compress_cache:
                    # 解压缩缓存，取前src_s个token（当前序列长度）
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # 直接取缓存的前src_s个token
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                # 更新缓存：将新生成的k/v添加到缓存末尾
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # 维度重排：适配bmm运算
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                # 计算注意力输出（根据设备类型适配精度）
                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    # CPU设备：转换为float32计算，再转回half精度
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # 稀疏注意力（仅关注部分历史token）
                # 更新k缓存（v缓存通过稀疏索引获取）
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                # 计算稀疏注意力输出
                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim, attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim, attn_sparsity).cuda().half()
        else:  # 混合设备缓存（部分在GPU、部分在CPU/磁盘）
            assert attn_sparsity >= 1.0  # 混合设备仅支持稠密注意力
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s, n_head, head_dim)

        # 5. 输出投影与残差连接
        value = value.transpose(1, 2).view(b, tgt_s, h)  # 恢复形状
        value = F.linear(value, w_out.data, bias=b_out.data)  # 输出投影
        value.add_(inputs.data)  # 残差连接

        # 释放输入张量内存（若允许）
        if donate[0]:
            inputs.delete()
        if donate[1]:
            attention_mask.delete()

        # 6. 更新k/v缓存（压缩或直接存储）
        if compress_cache:
            # 按压缩配置裁剪缓存（按分组大小对齐）
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            # 压缩新生成的k/v
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            # 直接封装为TorchTensor
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        # 返回输出和更新后的缓存
        return TorchTensor.create_from_torch(value, self), k_new, v_new
    
    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        """
        辅助函数：计算注意力权重（生成阶段专用）
        Args:
            q: 查询张量（shape: (b×n_head, 1, head_dim)）
            k: 键张量（shape: (b×n_head, head_dim, src_s)）
            mask: 注意力掩码（shape: (b, src_s)）
            b: 批大小
            src_s: 当前序列长度（输入+已生成token）
            n_head: 注意力头数
        Returns:
            归一化后的注意力权重（shape: (b×n_head, 1, src_s)）
        """
        # 1. 计算注意力分数：q × k^T（batch matrix multiply）
        # shape: (b × n_head, 1, src_s)
        attn_weights = torch.bmm(q, k)
        # 2. 调整掩码形状：适配注意力权重的维度
        # shape: (b, 1, 1, src_s)
        mask = mask.view(b, 1, 1, src_s)
        # 3. 掩码处理：恢复头维度 → 应用掩码 → 重排为批量格式
        attn_weights = attn_weights.view(b, n_head, 1, src_s)  # (b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)  # 掩码位置设为极小值（不参与softmax）
        attn_weights = attn_weights.view(b * n_head, 1, src_s)  # (b×n_head, 1, src_s)
        # 4. 权重归一化：softmax确保权重和为1
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        """
        辅助函数：计算注意力输出（稠密注意力）
        Args:
            q: 查询张量（shape: (b×n_head, tgt_s, head_dim)）
            k: 键张量（shape: (b×n_head, head_dim, src_s)）
            v: 值张量（shape: (b×n_head, src_s, head_dim)）
            mask: 注意力掩码（shape: (b, src_s)）
            b: 批大小
            src_s: 源序列长度（历史token数）
            tgt_s: 目标序列长度（当前生成token数，生成阶段为1）
            n_head: 注意力头数
            head_dim: 每个头的维度
        Returns:
            注意力输出（shape: (b, n_head, tgt_s, head_dim)）
        """
        # 1. 计算注意力权重
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # 2. 计算注意力输出：权重 × v，再恢复原始维度
        # shape: (b×n_head, tgt_s, head_dim) → (b, n_head, tgt_s, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        """
        稀疏注意力输出计算：仅关注部分历史token（降低计算量和显存占用）
        Args:
            q: 查询张量（shape: (b×n_head, 1, head_dim)）
            k: 键张量（shape: (b×n_head, head_dim, src_s)）
            v_new: 当前token的v张量（shape: (1, b×n_head, head_dim)）
            v_cache: 历史v缓存（GPU/CPU设备上的张量）
            mask: 注意力掩码（shape: (b, src_s)）
            b: 批大小
            src_s: 当前序列长度
            tgt_s: 目标序列长度（生成阶段为1）
            n_head: 注意力头数
            head_dim: 每个头的维度
            attn_sparsity: 稀疏度（0~1，如0.1表示仅关注10%的历史token）
        Returns:
            稀疏注意力输出（shape: (b, n_head, tgt_s, head_dim)）
        """
        # 1. 计算注意力权重（稠密格式）
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # 2. 筛选Top-K权重：仅保留权重最大的topk个历史token（排除当前token）
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))  # 排除最后一个当前token
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)  # 取topk个权重和对应的索引
        # 3. 调整索引形状：适配缓存读取（shape: (topk, b×n_head)）
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # 4. 拼接权重：topk权重 + 当前token权重（最后一维）
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)  # (b×n_head, 1, topk+1)

        # 5. 准备v缓存：根据设备类型分配临时缓冲区
        if k.is_cuda:
            v_home = v_cache  # GPU设备：v_cache是原始缓存
            # 分配临时缓冲区（存储稀疏筛选后的v张量）
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()  # 索引转CPU处理
        else:
            (v_home, v_buf) = v_cache  # CPU设备：v_cache是（原始缓存, 临时缓冲区）

        # 6. 稀疏读取v缓存：从历史缓存中读取topk个token的v值
        indices_src = topk_indices  # 源索引（历史缓存中的topk位置）
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))  # 目标索引（缓冲区位置）
        general_copy(v_buf, indices_tgt, v_home, indices_src)  # 异步拷贝
        v_home.device.synchronize()  # 等待拷贝完成

        # 7. 更新v缓冲区：将当前token的v_new添加到缓冲区末尾
        v = v_buf.data[:topk+1]  # (topk+1, b×n_head, head_dim)
        v[topk:topk+1] = v_new  # 最后一行写入当前token的v值
        # 8. 维度重排：适配注意力权重乘法（shape: (b×n_head, topk+1, head_dim)）
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # 9. 计算稀疏注意力输出并恢复维度
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        """
        混合设备注意力计算：缓存拆分到GPU和CPU，分别计算后拼接结果
        核心作用：突破单设备显存限制，利用CPU内存扩展缓存容量
        Args:
            q: 查询张量（shape: (b×n_head, 1, head_dim)）
            k_cache/v_cache: 混合设备缓存（(GPU缓存, CPU缓存)）
            k_new/v_new: 当前token的k/v张量（shape: (1, b×n_head, head_dim)）
            mask: 注意力掩码（shape: (b, src_s)）
            b: 批大小
            src_s: 当前序列长度
            tgt_s: 目标序列长度（生成阶段为1）
            n_head: 注意力头数
            head_dim: 每个头的维度
        Returns:
            混合设备注意力输出（shape: (b, n_head, tgt_s, head_dim)）
        """
        # 1. 拆分混合设备缓存：GPU部分和CPU部分
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]  # 拆分边界（GPU缓存的第二维长度）

        # 2. 计算GPU部分注意力
        b_gpu = seg // n_head  # GPU处理的批大小（按头数拆分）
        q_gpu = q[:seg]  # GPU对应的q张量（前seg个元素）
        # 更新GPU缓存：取前src_s个token，末尾添加当前token的k/v
        k_gpu = k_gpu[:src_s, :seg, :]  # (src_s, seg, head_dim)
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]  # 添加当前k
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]  # 添加当前v
        # 维度重排：适配bmm运算
        k_gpu = k_gpu.permute(1, 2, 0)  # (seg, head_dim, src_s)
        v_gpu = v_gpu.permute(1, 0, 2)  # (seg, src_s, head_dim)
        # 掩码适配GPU设备
        mask_gpu = mask[:b_gpu].cuda()
        # 计算GPU部分注意力输出
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # 3. 计算CPU部分注意力
        b_cpu = b - b_gpu  # CPU处理的批大小（剩余部分）
        q_cpu = q[seg:].float().cpu()  # CPU对应的q张量（转float32计算）
        # 更新CPU缓存：取前src_s个token，末尾添加当前token的k/v
        k_cpu = k_cpu[:src_s, seg:, :]  # (src_s, b×n_head - seg, head_dim)
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]  # 添加当前k
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]  # 添加当前v
        # 维度重排：适配bmm运算
        k_cpu = k_cpu.permute(1, 2, 0)  # (b×n_head - seg, head_dim, src_s)
        v_cpu = v_cpu.permute(1, 0, 2)  # (b×n_head - seg, src_s, head_dim)
        # CPU对应的掩码
        mask_cpu = mask[b_gpu:]
        # 计算CPU部分注意力输出
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        # 4. 拼接结果：CPU结果转GPU半精度，与GPU结果合并
        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        """
        OPT模型的MLP层（前馈网络）：包含层归一化、线性投影、ReLU激活和残差连接
        Args:
            inputs: 输入张量（shape: (batch_size, seq_len, hidden_dim)）
            wi: 中间层线性投影权重（shape: (hidden_dim, 4×hidden_dim)）
            bi: 中间层偏置（shape: (4×hidden_dim,)）
            wo: 输出层线性投影权重（shape: (4×hidden_dim, hidden_dim)）
            bo: 输出层偏置（shape: (hidden_dim,)）
            w_ln/b_ln: 层归一化权重/偏置（shape: (hidden_dim,)）
            donate: 是否释放输入张量内存（[True/False]）
        Returns:
            MLP层输出（shape: (batch_size, seq_len, hidden_dim)）
        """
        # 若权重是压缩的，先解压缩
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape  # batch_size, seq_len, hidden_dim

        # 1. 层归一化
        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        # 2. 中间层线性投影（hidden_dim → 4×hidden_dim）
        out = F.linear(out, wi.data, bias=bi.data)
        # 3. ReLU激活函数（原地操作节省内存）
        F.relu(out, inplace=True)
        # 4. 输出层线性投影（4×hidden_dim → hidden_dim）
        out = F.linear(out, wo.data, bias=bo.data)

        # 5. 残差连接：加上原始输入（归一化前）
        out.add_(inputs.data)
        # 释放输入张量内存（若允许）
        if donate[0]:
            inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        """设备同步：等待当前设备上所有异步操作完成（仅GPU有效）"""
        torch.cuda.synchronize()

    def mem_stats(self):
        """
        获取设备内存使用统计
        Returns:
            (当前内存占用, 峰值内存占用)：单位为字节
        """
        if self.device_type == DeviceType.CUDA:
            # GPU设备：使用PyTorch的CUDA内存统计
            cur_mem = torch.cuda.memory_allocated(self.dev)  # 当前已分配内存
            peak_mem = torch.cuda.max_memory_allocated(self.dev)  # 峰值内存
        elif self.device_type == DeviceType.CPU:
            # CPU设备：使用工具函数统计内存
            cur_mem = cpu_mem_stats()
            peak_mem = 0  # CPU暂不统计峰值内存
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        """
        打印设备内存使用统计（单位：GB）
        Args:
            output_file: 输出文件路径（None则打印到控制台）
        Returns:
            (当前内存占用, 峰值内存占用)：单位为字节
        """
        torch.cuda.synchronize()  # 确保所有GPU操作完成，统计准确
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            # 写入文件
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            # 打印到控制台
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        """设备字符串表示（便于调试）"""
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """磁盘设备管理类：负责张量的磁盘存储、异步拷贝和生命周期管理
    核心作用：作为Offloading存储介质，存储GPU/CPU放不下的权重或缓存
    """

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        """
        初始化磁盘设备
        Args:
            path: 磁盘缓存目录路径
            mem_capacity: 磁盘存储容量限制（可选）
            cuda_id: 关联的GPU ID（用于拷贝时的设备适配）
            num_copy_threads: 异步拷贝线程数（默认4个）
        """
        self.name = path  # 设备名称（即磁盘目录路径）
        self.path = os.path.abspath(os.path.expanduser(path))  # 绝对路径
        self.mem_capacity = mem_capacity  # 存储容量限制

        self.device_type = DeviceType.DISK  # 设备类型为磁盘
        self.compressed_device = TorchCompressedDevice(self)  # 关联的压缩设备

        # 校验目录：存在则确保是目录，不存在则创建
        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}  # 设备间连接（用于IO优化）

        # 初始化异步拷贝线程
        self.copy_queue = queue.Queue()  # 拷贝任务队列
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        # 启动所有拷贝线程
        for t in self.copy_threads:
            t.start()

        # 设置为全局磁盘设备（供其他模块调用）
        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        """添加设备间连接（用于IO传输优化）"""
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        """
        在磁盘上分配张量存储（使用numpy memmap实现内存映射）
        Args:
            shape: 张量形状
            dtype: 数据类型（numpy格式）
            pin_memory: 锁页内存标记（磁盘设备无效，忽略）
            name: 张量名称（可选，默认自动生成）
        Returns:
            磁盘上的TorchTensor实例（data为文件路径）
        """
        name = name or TorchTensor.next_name()  # 生成唯一名称
        path = os.path.join(self.path, name)  # 张量文件路径
        # 创建内存映射文件（延迟加载，不占用物理内存）
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        # 返回TorchTensor实例（data为文件路径）
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        """删除磁盘上的张量文件（释放磁盘空间）"""
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        """
        初始化单GPU批处理的磁盘缓存（k/v缓存）
        Args:
            config: 模型配置（n_head, input_dim）
            task: 推理任务配置（prompt_len, gen_len）
            policy: 推理策略（gpu_batch_size）
        Returns:
            (k_cache, v_cache)：磁盘上的k/v缓存张量
        """
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        # 缓存形状：(最大序列长度, 批大小×头数, 头维度)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # 分配磁盘缓存（float16类型，节省磁盘空间）
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def submit_copy(self, *args):
        """提交拷贝任务到异步队列（非阻塞）"""
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        """等待所有异步拷贝任务完成（阻塞）"""
        self.copy_queue.join()

    def close_copy_threads(self):
        """关闭所有异步拷贝线程（程序退出时调用）"""
        # 向队列发送None信号，通知线程退出
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        # 等待所有线程结束
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        """磁盘设备暂不实现内存统计"""
        raise NotImplementedError()

    def print_stats(self):
        """磁盘设备暂不实现统计打印"""
        raise NotImplementedError()

    def __del__(self):
        """析构函数：关闭拷贝线程（避免资源泄漏）"""
        if self.copy_queue:
            self.close_copy_threads()


# 混合设备的张量分段维度（默认按第二维拆分，对应batch×n_head维度）
SEG_DIM = 1

class TorchMixedDevice:
    """混合设备管理类：管理分布在多个物理设备（GPU/CPU/磁盘）上的张量
    核心逻辑：按指定维度（SEG_DIM）将张量拆分为多个段，分别存储在不同设备
    """

    def __init__(self, base_devices):
        """
        初始化混合设备
        Args:
            base_devices: 基础设备列表（如[GPU设备, CPU设备, 磁盘设备]）
        """
        self.name = "mixed"  # 设备名称
        self.device_type = DeviceType.MIXED  # 设备类型为混合设备
        self.base_devices = base_devices  # 基础物理设备列表

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        """
        在混合设备上分配张量：按seg_lengths拆分张量到各个基础设备
        Args:
            shape: 完整张量形状
            dtype: 数据类型（numpy格式）
            seg_lengths: 各设备的分段长度列表（总和等于shape[SEG_DIM]）
            pin_memory: 锁页内存标记（传递给基础设备）
            name: 张量名称（可选）
        Returns:
            混合设备上的TorchTensor实例（data为(子张量元组, 分段点元组)）
        """
        # 校验：分段长度总和必须等于分段维度的大小
        assert sum(seg_lengths) == shape[SEG_DIM]
        # 校验：分段长度数量必须等于基础设备数量
        assert len(seg_lengths) == len(self.base_devices)

        # 计算分段点（如seg_lengths=[1024,2048] → seg_points=[0,1024,3072]）
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []  # 存储各设备上的子张量
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]  # 当前分段的长度
            if seg_len == 0:
                tensors.append(None)  # 长度为0则存None
            else:
                # 计算当前分段的形状（仅分段维度变化）
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                # 在基础设备上分配子张量
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        # 返回混合设备张量（data为(子张量元组, 分段点元组)）
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        """删除混合设备上的所有子张量（递归释放各设备资源）"""
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy):
        """
        初始化混合设备的k/v缓存：按策略分配GPU/CPU/磁盘的缓存比例
        Args:
            config: 模型配置（n_head, input_dim）
            task: 推理任务配置（prompt_len, gen_len）
            policy: 推理策略（cache_gpu_percent/cache_cpu_percent/cache_disk_percent）
        Returns:
            (k_cache, v_cache)：混合设备上的k/v缓存张量
        """
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        # 缓存完整形状
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # 计算各设备的缓存分段长度（必须是n_head的整数倍，确保头维度对齐）
        if policy.cache_disk_percent == 0:
            # 无磁盘缓存：仅GPU+CPU
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM] - len_gpu
            len_disk = 0
        else:
            # 有磁盘缓存：GPU+CPU+磁盘
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]  # 各设备分段长度

        pin_memory = False  # 禁用锁页内存（减少开销）
        # 分配混合设备的k/v缓存
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """设备间IO链路类：定义两个设备之间的传输带宽，用于性能估算"""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        """
        初始化IO链路
        Args:
            a/b: 两个连接的设备（TorchDevice/TorchDisk等）
            a_to_b_bandwidth: a→b的传输带宽（字节/秒）
            b_to_a_bandwidth: b→a的传输带宽（字节/秒）
        """
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth  # a到b的带宽
        self.b_to_a_bandwidth = b_to_a_bandwidth  # b到a的带宽

        # 将链路添加到两个设备的连接列表
        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        """
        计算IO传输时间（基于带宽和数据大小）
        Args:
            src: 源设备
            dst: 目标设备
            size: 传输数据大小（字节）
        Returns:
            传输时间（秒）
        """
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"无效的源设备 {src}")

        # 强制IO时间（用于测试，全局变量）
        if force_io_time is not None:
            return force_io_time

        # 时间 = 数据大小 / 带宽
        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """
    通用异步拷贝函数：支持任意设备组合（GPU/CPU/磁盘/混合/压缩）的张量拷贝
    功能等价于numpy语法：dst[dst_indices] = src[src_indices]
    注意：拷贝是异步的，需调用设备的synchronize()等待完成
    >>> env.disk.synchronize()  # 等待磁盘拷贝完成
    >>> torch.cuda.synchronize()  # 等待GPU拷贝完成
    """
    if dst.device.device_type == DeviceType.MIXED:
        # 目标是混合设备：递归拷贝到各子设备
        assert src.device.device_type != DeviceType.MIXED  # 不支持混合→混合拷贝
        seg_points = dst.data[1]  # 目标设备的分段点

        for i in range(len(dst.device.base_devices)):
            # 跳过长度为0的分段
            if seg_points[i] == seg_points[i+1]:
                continue
            # 补全默认索引（无索引则取整个张量）
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            # 切割索引：仅保留当前分段的范围
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            # 递归拷贝到当前子设备
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        # 源是混合设备：递归从各子设备拷贝
        assert dst.device.device_type != DeviceType.MIXED  # 不支持混合→混合拷贝
        seg_points = src.data[1]  # 源设备的分段点

        for i in range(len(src.device.base_devices)):
            # 跳过长度为0的分段
            if seg_points[i] == seg_points[i+1]:
                continue
            # 补全默认索引
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            # 切割索引：仅保留当前分段的范围
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            # 递归从当前子设备拷贝
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # 涉及压缩设备：调用压缩拷贝函数（递归处理解压缩/压缩）
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # 源是磁盘设备：提交到异步拷贝线程
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # 目标是磁盘设备：提交到异步拷贝线程
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # GPU→CPU且CPU张量未锁页：用磁盘设备的拷贝线程+锁页缓存中继
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # CPU→GPU且CPU张量未锁页：先转锁页内存再异步拷贝
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()  # 转为锁页内存（加速GPU拷贝）
        dst.copy_(src, non_blocking=True)  # 非阻塞拷贝
    else:
        # 常规路径：CPU-CPU、GPU-GPU、GPU-CPU（锁页）等
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)  # 异步拷贝


def cut_indices(indices, start, stop, base=0):
    """
    辅助函数：切割索引，仅保留分段维度上[start, stop)范围内的部分
    Args:
        indices: 原始索引元组（每个元素是slice对象）
        start: 分段起始位置
        stop: 分段结束位置
        base: 基础偏移量（用于调整目标索引的起始点）
    Returns:
        切割后的索引元组
    """
    # 校验：所有索引必须是连续的（无步长）
    assert all(x.step is None for x in indices)
    # 获取分段维度的原始索引
    seg = indices[SEG_DIM]
    # 切割分段维度索引：max(原始起始, 分段起始) → min(原始结束, 分段结束)，并减去基础偏移
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    """
    将TorchTensor映射为PyTorch张量（支持磁盘内存映射和稀疏索引）
    Args:
        tensor: TorchTensor实例（可在任意设备）
        indices: 索引（slice元组或张量）
    Returns:
        PyTorch张量（磁盘张量返回memmap对象，支持索引操作）
    """
    if tensor.device.device_type == DeviceType.DISK:
        # 磁盘设备：打开内存映射文件（不加载到物理内存）
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        # 其他设备：直接获取原始数据
        data = tensor.data

    # 向后兼容：仅处理稀疏v缓存的索引（张量类型索引）
    if torch.is_tensor(indices):
        return vector_gather(data, indices)  # 稀疏收集（按索引取元素）
    return data[indices] if indices else data  # 常规索引或返回完整数据


def copy_worker_func(queue, cuda_id):
    """
    异步拷贝工作线程函数：处理磁盘与其他设备的拷贝任务
    核心逻辑：用锁页内存作为中继，优化GPU与磁盘之间的传输
    """
    torch.cuda.set_device(cuda_id)  # 设置线程关联的GPU

    # 分配锁页内存缓冲区（1GB，float16类型）：用于数据中转
    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()  # 创建独立的CUDA流（不阻塞主线程）

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()  # 从队列获取任务
            if item is None:
                # 收到退出信号，完成任务并返回
                queue.task_done()
                return

            # 解析任务参数：目标张量、目标索引、源张量、源索引
            dst, dst_indices, src, src_indices = item
            # 映射源/目标张量为PyTorch张量（支持磁盘memmap）
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # 涉及GPU的拷贝：用锁页内存作为中继（加速传输）
                size = np.prod(src_data.shape)  # 计算数据大小
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)  # 分配临时缓冲区
                tmp_cpu_buf.copy_(src_data)  # 源→中继缓冲区
                dst_data.copy_(tmp_cpu_buf)  # 中继缓冲区→目标
            else:
                # CPU/磁盘之间的拷贝：直接拷贝
                dst_data.copy_(src_data)

            queue.task_done()  # 标记任务完成