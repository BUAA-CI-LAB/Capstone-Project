"""
FlexLLMGen OPT模型推理核心模块（输入嵌入部分）
支持模型权重在GPU/CPU/磁盘间灵活分配，结合量化压缩、IO与计算重叠等优化策略
用法示例：
python3 -m flexllmgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

# 导入内部核心模块：压缩配置、模型配置、PyTorch后端、计时器、工具函数
from flexllmgen.compression import CompressionConfig
from flexllmgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexllmgen.timer import timers
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

fix_recursive_import()  # 修复模块循环导入问题

DUMMY_WEIGHT = "_DUMMY_"  # 基准测试标记：使用全1虚拟权重，避免下载真实模型权重


@dataclasses.dataclass(frozen=True)
class Policy:
    """
    推理策略配置类：封装所有推理优化策略和资源分配规则
    采用不可变设计（frozen=True），确保策略初始化后不被意外修改
    """
    # 批处理配置
    gpu_batch_size: int  # 单GPU处理的请求批大小
    num_gpu_batches: int  # GPU批处理总数（多批时启用IO-计算重叠）

    # 资源分配比例（百分比，总和需为100%）
    # 格式：[权重GPU占比, 权重CPU占比, 缓存GPU占比, 缓存CPU占比, 激活值GPU占比, 激活值CPU占比]
    w_gpu_percent: float    # 模型权重在GPU上的存储比例
    w_cpu_percent: float    # 模型权重在CPU上的存储比例
    cache_gpu_percent: float# KV缓存在GPU上的存储比例
    cache_cpu_percent: float# KV缓存在CPU上的存储比例
    act_gpu_percent: float  # 中间激活值在GPU上的存储比例
    act_cpu_percent: float  # 中间激活值在CPU上的存储比例

    # 优化开关
    overlap: bool  # 是否启用IO（权重/缓存加载）与计算重叠
    sep_layer: bool  # 是否将注意力层和MLP层拆分为独立层（并行调度优化）
    pin_weight: bool  # CPU上的权重是否使用锁页内存（加速GPU-CPU数据传输）
    cpu_cache_compute: bool  # 是否在CPU上执行注意力计算（缓解GPU算力压力）

    # 注意力优化参数
    attn_sparsity: float  # 注意力权重稀疏率（1.0为稠密注意力，<1.0启用稀疏优化）

    # 权重量化配置
    compress_weight: bool  # 是否启用权重量化压缩
    comp_weight_config: CompressionConfig  # 权重量化参数（组大小、量化位数等）

    # KV缓存量化配置
    compress_cache: bool  # 是否启用KV缓存量化压缩
    comp_cache_config: CompressionConfig  # 缓存量化参数

    # 衍生属性：计算各资源在磁盘上的存储比例（剩余比例自动分配给磁盘）
    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    """
    根据当前百分比位置，从候选列表中选择对应的资源/设备
    核心逻辑：基于累积比例的区间匹配（用于权重/缓存的设备分配）
    Args:
        cur_percent: 当前资源的百分比位置（0~100）
        percents: 候选选项的比例分布（如[30,50,20]表示三个选项占比30%/50%/20%）
        choices: 与比例对应的候选列表（如[磁盘, CPU, GPU]）
    Returns:
        匹配的候选选项
    """
    percents = np.cumsum(percents)  # 计算累积比例（如[30,80,100]）
    assert np.abs(percents[-1] - 100) < 1e-5, "资源分配比例总和必须为100%"

    # 匹配当前百分比所在的累积区间
    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]  # 兜底返回最后一个选项（避免边界值遗漏）


def init_weight_list(weight_specs, policy, env):
    """
    初始化模型权重列表：根据推理策略分配权重到目标设备（磁盘/CPU/GPU）
    支持量化压缩和虚拟权重（基准测试用），按权重大小优先级分配到高速设备
    Args:
        weight_specs: 权重规格列表，每个元素为(shape, dtype, 权重文件路径)
        policy: 推理策略（含资源分配比例和量化配置）
        env: 执行环境（封装磁盘、CPU、GPU等设备实例）
    Returns:
        初始化后的权重张量列表（TorchTensor实例）
    """
    # 定义设备分配规则：比例顺序[磁盘, CPU, GPU]，对应设备实例
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    # 计算每个权重的元素总数（用于按大小排序分配）
    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)  # 权重大小累积和（用于计算百分比位置）
    ret = []

    for i in range(len(weight_specs)):
        shape, dtype, filename = weight_specs[i]
        # 计算当前权重的"中间百分比"（权重中心在总大小中的位置，确保公平分配）
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1] * 100
        # 根据中间百分比选择权重的归属设备（home设备）
        home = get_choice(mid_percent, dev_percents, dev_choices)

        # 配置权重存储参数：1D张量（偏置、层归一化权重）不量化，默认启用锁页内存
        if len(shape) < 2:
            pin_memory = True  # 1D张量体积小，锁页内存开销可忽略
            compress = False   # 1D张量量化收益低，且可能损失精度
        else:
            pin_memory = policy.pin_weight  # 按策略配置锁页内存
            compress = policy.compress_weight  # 按策略启用权重量化

        # 分配权重张量（支持量化和非量化两种模式）
        if not compress:
            # 非量化模式：直接在目标设备上分配张量
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)
            # 加载权重数据（虚拟权重或真实权重文件）
            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(filename)  # 从.npy文件加载真实权重
            else:
                weight.load_from_np(np.ones(shape, dtype))  # 加载全1虚拟权重
        else:
            # 量化模式：在目标设备的压缩子设备上分配张量
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)
            # 加载量化权重（数据+缩放因子）
            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(filename)
            else:
                # 虚拟量化权重：数据和缩放因子均填充为1
                for x in weight.data[:2]:  # data[0] = 量化数据, data[1] = 缩放因子
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    """
    输入嵌入模块：将token ID和位置信息转换为模型输入嵌入向量
    核心功能：token嵌入 + 位置嵌入 + 注意力掩码处理，输出嵌入张量
    """
    def __init__(self, config: OptConfig, env: ExecutionEnv, policy: Policy):
        """
        初始化输入嵌入模块
        Args:
            config: OPT模型配置（词汇表大小、嵌入维度等核心参数）
            env: 执行环境（封装GPU等计算设备）
            policy: 推理策略（含量化等优化配置）
        """
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu  # 嵌入计算固定在GPU上（性能最优）
        # 权重加载目标设备：量化时为GPU压缩设备，否则为GPU
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
                                else self.compute)
        self.task = None  # 推理任务实例（动态设置，含prompt长度、生成长度等）

    def set_task(self, task: Task):
        """
        设置当前推理任务
        Args:
            task: 推理任务实例（Task类，含prompt_len、gen_len、do_sample等参数）
        """
        self.task = task

    def init_weight(self, weight_home: ValueHolder, path: str):
        """
        初始化输入嵌入权重并存储到权重容器
        权重包括：token嵌入权重（vocab_size×input_dim）和位置嵌入权重（max_seq_len+2×input_dim）
        Args:
            weight_home: 权重存储容器（ValueHolder实例，用于统一管理权重生命周期）
            path: 权重文件根路径（.npy格式权重文件所在目录）
        """
        # 提取模型核心配置参数
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
                          self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")  # 确保路径以系统分隔符结尾（兼容Windows/Linux）
        # 定义权重规格：(形状, 数据类型, 相对路径)
        weight_specs = [
            # Token嵌入权重：将token ID映射为嵌入向量
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            # 位置嵌入权重：编码token在序列中的位置信息（+2为兼容预处理逻辑）
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]
        # 初始化权重并存储到容器
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home: ValueHolder, weight_read_buf: ValueHolder, k: int):
        """
        加载权重到读取缓冲区（仅在第0个GPU批处理时加载，复用权重）
        Args:
            weight_home: 权重存储容器（含token和位置嵌入权重）
            weight_read_buf: 权重读取缓冲区（计算时快速访问，避免重复加载）
            k: 当前GPU批处理索引（0~num_gpu_batches-1）
        """
        w_token, w_pos = weight_home.val  # 从容器中取出权重
        if k == 0:  # 仅在第一个批处理时加载（后续批处理复用缓冲区）
            # 智能拷贝：自动处理设备间传输（如CPU→GPU）
            weight_read_buf.store((w_token.smart_copy(self.weight_load_dst),
                                  w_pos.smart_copy(self.weight_load_dst)))

    def init_cache_one_gpu_batch(self, cache_home: ValueHolder):
        """
        初始化单GPU批处理的缓存（输入嵌入模块无KV缓存，空实现）
        Args:
            cache_home: 缓存存储容器（ValueHolder实例）
        """
        pass  # 输入嵌入无需缓存，直接返回

    def load_cache(self, cache_home: ValueHolder, cache_read_buf: ValueHolder, i: int):
        """
        加载缓存（输入嵌入模块无缓存，空实现）
        Args:
            cache_home: 缓存存储容器
            cache_read_buf: 缓存读取缓冲区
            i: 当前生成步骤索引（0~gen_len-1）
        """
        pass

    def store_cache(self, cache_home: ValueHolder, cache_write_buf: ValueHolder, i: int):
        """
        存储缓存（输入嵌入模块无缓存，空实现）
        Args:
            cache_home: 缓存存储容器
            cache_write_buf: 缓存写入缓冲区
            i: 当前生成步骤索引
        """
        pass

    def input_act_shape_and_dtype(self, batch_size: int, seq_len: int):
        """
        获取输入激活值的形状和数据类型（输入嵌入模块的输入为token ID）
        Args:
            batch_size: 批大小
            seq_len: 序列长度
        Returns:
            (shape, dtype): 输入形状为(batch_size, seq_len)，数据类型为np.int64（token ID类型）
        """
        return (batch_size, seq_len), np.int64

    def forward(self, hidden: ValueHolder, cache_read_buf: ValueHolder,
                weight_read_buf: ValueHolder, attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, i: int, k: int):
        """
        前向传播：计算输入嵌入（token嵌入 + 位置嵌入 + 掩码处理）
        Args:
            hidden: 隐藏状态容器（输入：token ID张量；输出：嵌入向量张量）
            cache_read_buf: 缓存读取缓冲区（未使用）
            weight_read_buf: 权重读取缓冲区（含token和位置嵌入权重）
            attention_mask: 注意力掩码容器（用于屏蔽padding token）
            cache_write_buf: 缓存写入缓冲区（未使用）
            i: 当前生成步骤索引（0为预填充阶段，>0为生成阶段）
            k: 当前GPU批处理索引
        """
        # 内存捐赠标记：标记可被覆盖的张量（优化GPU内存复用，减少显存占用）
        donate = [False] * 4
        h, donate[0] = hidden.val, True  # 输入token ID张量，标记为可捐赠（计算后可覆盖）
        # 将注意力掩码拷贝到计算设备（GPU），标记为可捐赠
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        # 读取权重：最后一个批处理时弹出缓冲区（释放内存），其他批处理直接复用
        if k == self.policy.num_gpu_batches - 1:
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        # 调用GPU后端的输入嵌入计算接口：token嵌入 + 位置嵌入 + 掩码应用
        h = self.compute.opt_input_embed(
            h, mask, w_token, w_pos, self.config.pad_token_id, donate
        )
        hidden.val = h  # 更新隐藏状态容器：嵌入向量张量（shape: [batch_size, seq_len, input_dim]）


class OutputEmbed:
    """
    输出嵌入模块：将模型最后一层隐藏状态转换为token概率分布
    核心流程：层归一化 → 输出投影（与输入嵌入共享权重） → 概率归一化（softmax）/采样
    支持贪心解码和带温度的随机采样
    """
    def __init__(self, config, env, policy):
        """
        初始化输出嵌入模块
        Args:
            config: OptConfig实例，模型核心配置（词汇表大小、输入维度等）
            env: ExecutionEnv实例，执行环境（封装GPU/CPU/磁盘设备）
            policy: Policy实例，推理策略（含量化、设备分配等配置）
        """
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu  # 输出计算固定在GPU上（性能最优）
        # 权重加载目标设备：量化时为GPU压缩设备，否则为GPU
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
                                else self.compute)
        self.task = None  # 推理任务实例（动态设置，含采样参数等）

    def set_task(self, task):
        """
        设置当前推理任务
        Args:
            task: Task实例，含生成长度、采样开关、温度等参数
        """
        self.task = task

    def init_weight(self, weight_home, path):
        """
        初始化输出嵌入权重并存储到权重容器
        权重组成：输出层归一化（weight+bias） + 输出投影权重（与输入token嵌入共享）
        Args:
            weight_home: ValueHolder实例，权重存储容器（管理权重生命周期）
            path: str，权重文件根路径（.npy格式文件所在目录）
        """
        # 提取模型核心参数：词汇表大小、输入维度、数据类型
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
                       self.config.dtype)
        path = os.path.join(path, "")  # 确保路径以系统分隔符结尾（兼容多平台）
        # 定义权重规格：(形状, 数据类型, 相对路径)
        weight_specs = [
            # 输出层归一化权重：用于最后一层隐藏状态的归一化
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            # 输出层归一化偏置
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            # 输出投影权重：与输入token嵌入共享（参数高效设计）
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        # 按策略初始化权重（分配设备、加载数据）并存储到容器
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        """
        加载权重到读取缓冲区（仅第0个GPU批处理时加载，复用权重）
        Args:
            weight_home: ValueHolder实例，权重存储容器
            weight_read_buf: ValueHolder实例，权重读取缓冲区（计算时快速访问）
            k: int，当前GPU批处理索引（0~num_gpu_batches-1）
        """
        # 从容器中取出权重：层归一化（w_ln/b_ln）、输出投影（w_token）
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:  # 仅第一个批处理加载（后续批处理复用缓冲区，减少IO）
            dst1 = self.weight_load_dst  # 量化权重目标设备（输出投影权重）
            dst2 = self.compute  # 非量化权重目标设备（层归一化权重/偏置）
            # 智能拷贝：自动处理设备间传输（如CPU→GPU、压缩格式转换）
            weight_read_buf.store((
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)
            ))

    def init_cache_one_gpu_batch(self, cache_home):
        """
        初始化单GPU批处理的缓存（输出嵌入模块无KV缓存，空实现）
        Args:
            cache_home: ValueHolder实例，缓存存储容器
        """
        pass  # 输出嵌入无需缓存，直接返回

    def load_cache(self, cache_home, cache_read_buf, i):
        """
        加载缓存（输出嵌入模块无缓存，空实现）
        Args:
            cache_home: ValueHolder实例，缓存存储容器
            cache_read_buf: ValueHolder实例，缓存读取缓冲区
            i: int，当前生成步骤索引（0~gen_len-1）
        """
        pass

    def store_cache(self, cache_home, cache_write_buf, i):
        """
        存储缓存（输出嵌入模块无缓存，空实现）
        Args:
            cache_home: ValueHolder实例，缓存存储容器
            cache_write_buf: ValueHolder实例，缓存写入缓冲区
            i: int，当前生成步骤索引
        """
        pass

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        """
        获取输入激活值的形状和数据类型（最后一层隐藏状态）
        Args:
            batch_size: int，批大小
            seq_len: int，序列长度（生成阶段为1，预填充阶段为prompt长度）
        Returns:
            tuple: ((batch_size, seq_len, input_dim), dtype)，输入形状和数据类型
        """
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        """
        前向传播：将最后一层隐藏状态转换为token概率分布（或采样结果）
        Args:
            hidden: ValueHolder实例，输入为最后一层隐藏状态，输出为token概率/采样结果
            cache_read_buf: ValueHolder实例，缓存读取缓冲区（未使用）
            weight_read_buf: ValueHolder实例，权重读取缓冲区（含层归一化和输出投影权重）
            attention_mask: ValueHolder实例，注意力掩码（未使用）
            cache_write_buf: ValueHolder实例，缓存写入缓冲区（未使用）
            i: int，当前生成步骤索引
            k: int，当前GPU批处理索引
        """
        # 内存捐赠标记：标记可被覆盖的张量（优化GPU内存复用，减少显存占用）
        donate = [False] * 4
        h, donate[0] = hidden.val, True  # 输入隐藏状态，标记为可捐赠

        # 读取权重：最后一个批处理时弹出缓冲区（释放内存），其他批处理复用
        if k == self.policy.num_gpu_batches - 1:
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        # 调用GPU后端的输出嵌入计算接口：层归一化→输出投影→softmax/采样
        h = self.compute.opt_output_embed(
            h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature  # 采样参数：是否采样、温度
        )
        hidden.val = h  # 更新隐藏状态：token概率分布（贪心解码）或采样后的token ID


class SelfAttention:
    """
    自注意力模块：实现多头自注意力计算，含KV缓存的初始化、加载、存储
    支持GPU/CPU混合计算、KV缓存量化、稀疏注意力、混合设备缓存等优化
    核心流程：层归一化→QKV投影→注意力计算→输出投影→残差连接
    """
    def __init__(self, config, env, policy, layer_id):
        """
        初始化自注意力模块
        Args:
            config: OptConfig实例，模型核心配置（输入维度、头数等）
            env: ExecutionEnv实例，执行环境（封装GPU/CPU/磁盘设备）
            policy: Policy实例，推理策略（含设备分配、量化、稀疏等配置）
            layer_id: int，当前自注意力层的索引（用于加载对应层权重）
        """
        self.config = config
        self.env = env
        self.layer_id = layer_id  # 层索引（区分不同自注意力层的权重）
        self.policy = policy
        self.compute = self.env.gpu  # 主计算设备（GPU，负责QKV投影、输出投影等）
        # 权重加载目标设备：量化时为GPU压缩设备，否则为GPU
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
                                else self.compute)
        # 注意力计算设备：按策略选择GPU（性能优先）或CPU（缓解GPU压力）
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
                                  else self.env.gpu)
        self.task = None  # 推理任务实例（动态设置，含prompt长度、生成长度等）

    def set_task(self, task):
        """
        设置当前推理任务
        Args:
            task: Task实例，含prompt_len、gen_len等参数
        """
        self.task = task

    def init_weight(self, weight_home, path):
        """
        初始化自注意力层权重并存储到权重容器
        权重组成：Q/K/V投影（weight+bias） + 输出投影（weight+bias） + 层归一化（weight+bias）
        Args:
            weight_home: ValueHolder实例，权重存储容器
            path: str，权重文件根路径
        """
        h, dtype = (self.config.input_dim, self.config.dtype)
        # 构建当前层的权重路径：decoder.layers.{layer_id}.self_attn
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        # 定义权重规格：(形状, 数据类型, 相对路径)
        weight_specs = [
            ((h, h), dtype, path + ".q_proj.weight"),  # Q投影权重（查询向量）
            ((h,), dtype, path + ".q_proj.bias"),      # Q投影偏置
            ((h, h), dtype, path + ".k_proj.weight"),  # K投影权重（键向量）
            ((h,), dtype, path + ".k_proj.bias"),      # K投影偏置
            ((h, h), dtype, path + ".v_proj.weight"),  # V投影权重（值向量）
            ((h,), dtype, path + ".v_proj.bias"),      # V投影偏置
            ((h, h), dtype, path + ".out_proj.weight"),# 输出投影权重（注意力结果融合）
            ((h,), dtype, path + ".out_proj.bias"),    # 输出投影偏置
            ((h,), dtype, path + "_layer_norm.weight"),# 自注意力层归一化权重
            ((h,), dtype, path + "_layer_norm.bias"),  # 自注意力层归一化偏置
        ]
        # 按策略初始化权重（分配设备、加载数据）并存储到容器
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        """
        加载权重到读取缓冲区（仅第0个GPU批处理时加载，复用权重）
        Args:
            weight_home: ValueHolder实例，权重存储容器（含10个权重）
            weight_read_buf: ValueHolder实例，权重读取缓冲区
            k: int，当前GPU批处理索引
        """
        # 从容器中取出所有权重
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:  # 仅第一个批处理加载（后续批处理复用，减少IO开销）
            dst1 = self.weight_load_dst  # 量化权重目标设备（Q/K/V/输出投影权重）
            dst2 = self.compute  # 非量化权重目标设备（偏置、层归一化权重/偏置）
            # 智能拷贝到目标设备，存储到缓冲区
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)
            ))

    def init_cache_one_gpu_batch(self, cache_home):
        """
        初始化单GPU批处理的KV缓存（K和V分别存储）
        根据策略选择缓存设备（GPU/CPU/磁盘/混合），支持量化压缩
        Args:
            cache_home: ValueHolder实例，缓存存储容器（用于存储K和V缓存）
        """
        # 按缓存分配比例选择目标设备
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu  # 全量GPU缓存（性能最优）
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu  # 全量CPU缓存（缓解GPU显存压力）
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk  # 全量磁盘缓存（极致显存节省，性能最差）
        else:
            device = self.env.mixed  # 混合设备缓存（部分GPU+部分CPU/磁盘）

        # 若启用缓存量化，切换到设备的压缩子设备（混合设备不支持量化）
        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED, "混合设备不支持KV缓存量化"
            device = device.compressed_device

        # 初始化KV缓存（形状：(max_seq_len, batch_size×n_head, head_dim)）
        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)  # 存储到缓存容器

    def load_cache(self, cache_home, cache_read_buf, i):
        """
        加载KV缓存到读取缓冲区（生成阶段使用，预填充阶段i=0不加载）
        根据设备类型和策略选择不同加载路径（直接拷贝/CPU缓冲区/混合加载）
        Args:
            cache_home: ValueHolder实例，缓存存储容器（含K和V缓存）
            cache_read_buf: ValueHolder实例，缓存读取缓冲区（供注意力计算使用）
            i: int，当前生成步骤索引（i=0为预填充，不加载缓存）
        """
        if i == 0:  # 预填充阶段：无历史上下文，无需加载缓存
            return

        k_home, v_home = cache_home.val  # 从容器中取出KV缓存

        # 选择缓存加载路径（基于量化、计算设备、缓存设备类型）
        if self.policy.compress_cache:
            # 量化缓存：直接加载到注意力计算设备的压缩子设备
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                # CPU计算注意力：判断是否为混合设备缓存
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2  # 混合加载（GPU部分直接用，CPU部分拷贝到缓冲区）
                else:
                    path = 1  # 拷贝到CPU临时缓冲区（统一计算）
            else:
                path = 0  # GPU计算注意力：直接拷贝到GPU（性能最优）
            dst = self.attention_compute

        # 执行缓存加载（不同路径对应不同拷贝逻辑）
        if path == 0:  # 路径0：直接拷贝到计算设备
            # 缓存索引：取前(prompt_len + i)个序列位置（包含历史上下文）
            indices = (slice(0, self.task.prompt_len + i), slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:  # 稠密注意力：加载K和V缓存
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),  # K缓存拷贝到目标设备
                    v_home.smart_copy(dst, indices),  # V缓存拷贝到目标设备
                ))
            else:  # 稀疏注意力：仅加载K缓存（V缓存按需加载，节省内存）
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),  # 标记V缓存暂不加载
                ))
        elif path == 1:  # 路径1：拷贝到CPU临时缓冲区（CPU计算注意力）
            # 获取CPU预分配的注意力计算工作空间（K和V缓冲区）
            k_buf, v_buf = dst.next_attention_compute_workspace()
            # 索引：取前(prompt_len + i - 1)个位置（排除当前生成步骤）
            indices = (slice(0, self.task.prompt_len + i - 1), slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)  # 拷贝K缓存到CPU缓冲区

            if self.policy.attn_sparsity >= 1.0:  # 稠密注意力：拷贝V缓存
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))  # 存储缓冲区引用
            else:  # 稀疏注意力：存储V缓存和缓冲区引用（按需加载）
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # 路径2：混合加载（GPU+CPU，混合设备缓存）
            # 混合设备缓存：GPU部分直接使用，CPU部分拷贝到CPU缓冲区
            gpu_k_buf = k_home.data[0][0]  # GPU上的K缓存片段
            gpu_v_buf = v_home.data[0][0]  # GPU上的V缓存片段

            # 获取CPU工作空间，拷贝CPU/磁盘上的缓存片段
            k_buf, v_buf = dst.next_attention_compute_workspace()
            # 索引：CPU部分的缓存（从GPU片段长度开始）
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)  # 拷贝K缓存到CPU
            general_copy(v_buf, indices, v_home, indices)  # 拷贝V缓存到CPU

            # 存储混合缓存（GPU片段 + CPU片段）
            cache_read_buf.store((((gpu_k_buf, k_buf), False),
                                  ((gpu_v_buf, v_buf), False)))
            assert self.policy.attn_sparsity >= 1.0, "混合加载不支持稀疏注意力"
        else:
            raise ValueError(f"无效的缓存加载路径: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        """
        存储当前步骤的KV缓存到缓存容器（预填充和生成阶段均需存储）
        最后一个生成步骤无需存储（无后续步骤使用）
        Args:
            cache_home: ValueHolder实例，缓存存储容器（目标位置）
            cache_write_buf: ValueHolder实例，缓存写入缓冲区（含当前步骤的K和V）
            i: int，当前生成步骤索引
        """
        # 取出目标缓存和当前步骤的KV（弹出缓冲区，释放临时内存）
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # 最后一个生成步骤：无需存储（无后续使用）
            return

        # 确定存储索引：预填充阶段存储整个prompt，生成阶段存储当前步骤
        if i == 0:  # 预填充阶段：存储[0, prompt_len)所有位置
            indices = (slice(0, k_new.shape[0]), slice(0, k_new.shape[1]))
        else:  # 生成阶段：存储当前步骤位置（prompt_len + i）
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos), slice(0, k_new.shape[1]))

        # 存储当前KV到缓存容器（覆盖对应位置）
        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        """
        获取输入激活值的形状和数据类型（上一层输出的隐藏状态）
        Args:
            batch_size: int，批大小
            seq_len: int，序列长度
        Returns:
            tuple: ((batch_size, seq_len, input_dim), dtype)，输入形状和数据类型
        """
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        """
        前向传播：执行多头自注意力计算（分预填充和生成两个阶段）
        核心流程：层归一化 → QKV投影 → 注意力计算 → 输出投影 → 残差连接
        Args:
            hidden: ValueHolder实例，输入为上一层隐藏状态，输出为自注意力层结果
            cache_read_buf: ValueHolder实例，缓存读取缓冲区（含历史KV缓存）
            weight_read_buf: ValueHolder实例，权重读取缓冲区（含自注意力层所有权重）
            attention_mask: ValueHolder实例，注意力掩码（屏蔽padding和未来位置）
            cache_write_buf: ValueHolder实例，缓存写入缓冲区（存储当前步骤的KV）
            i: int，当前生成步骤索引（0为预填充，>0为生成）
            k: int，当前GPU批处理索引
        """
        n_head = self.config.n_head  # 注意力头数
        # 内存捐赠标记：优化GPU内存复用（标记可被覆盖的张量）
        donate = [False] * 14
        h, donate[0] = hidden.val, True  # 输入隐藏状态，标记为可捐赠

        # 读取权重：最后一个批处理时弹出缓冲区（释放内存），其他批处理复用
        if k == self.policy.num_gpu_batches - 1:
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        if i == 0:  # 预填充阶段（处理整个prompt，无历史缓存）
            # 拷贝注意力掩码到计算设备（GPU）
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            # 调用GPU后端预填充注意力接口：层归一化→QKV投影→注意力计算→输出投影→残差连接
            h, new_k_cache, new_v_cache = self.compute.mha(
                h, mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out,
                w_ln, b_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config
            )
            # 存储当前步骤的KV到写入缓冲区（预填充阶段为整个prompt的KV）
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # 生成阶段（处理单个token，使用历史缓存）
            # 拷贝注意力掩码到注意力计算设备（GPU/CPU）
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            # 取出历史KV缓存（从读取缓冲区弹出）
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            # 调用GPU后端生成阶段注意力接口：复用历史缓存，仅计算当前token的注意力
            h, new_k_cache, new_v_cache = self.compute.mha_gen(
                h, mask, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out,
                w_ln, b_ln, n_head, k_cache, v_cache, donate,
                self.policy.attn_sparsity, self.policy.compress_cache,
                self.policy.comp_cache_config
            )
            # 存储当前步骤的KV到写入缓冲区（单个token的KV）
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h  # 更新隐藏状态：自注意力层输出（含残差连接）


class MLP:
    """
    前馈网络（MLP）模块：Transformer层核心组件，实现特征的非线性变换
    核心流程：层归一化 → 高维投影（h→4h） → GELU激活 → 低维投影（4h→h） → 残差连接
    适配量化策略和多设备权重存储
    """
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu  # MLP计算默认绑定GPU（性能最优）
        # 权重加载目标设备：量化时指向GPU压缩设备，否则直接使用GPU
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
                                else self.compute)
        self.task = None  # 推理任务实例（动态绑定）

    def set_task(self, task):
        """绑定推理任务（含生成长度、采样参数等）"""
        self.task = task

    def init_weight(self, weight_home, path):
        """
        初始化MLP层权重：fc1（升维）、fc2（降维）、层归一化相关权重
        Args:
            weight_home: ValueHolder，权重存储容器
            path: 权重文件根路径（decoder.layers.{layer_id}. 层级）
        """
        h, dtype = (self.config.input_dim, self.config.dtype)
        # 构建当前层权重路径（拼接Transformer层索引）
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        # 定义权重规格：(形状, 数据类型, 文件名)
        weight_specs = [
            ((4 * h, h), dtype, path + "fc1.weight"),  # 升维投影权重（h→4h）
            ((4 * h,), dtype, path + "fc1.bias"),      # 升维偏置
            ((h, 4 * h), dtype, path + "fc2.weight"),  # 降维投影权重（4h→h）
            ((h,), dtype, path + "fc2.bias"),          # 降维偏置
            ((h,), dtype, path + "final_layer_norm.weight"),  # 层归一化权重
            ((h,), dtype, path + "final_layer_norm.bias"),    # 层归一化偏置
        ]
        # 按策略分配权重设备并加载数据
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        """
        加载权重到缓冲区（仅第0个GPU批处理时初始化，复用权重）
        Args:
            weight_home: 权重存储容器
            weight_read_buf: 权重读取缓冲区
            k: 当前GPU批处理索引
        """
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if k == 0:  # 仅首次批处理加载（避免重复IO）
            dst1 = self.weight_load_dst  # 量化权重目标设备（fc1/fc2权重）
            dst2 = self.compute  # 非量化权重目标设备（偏置、归一化参数）
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)
            ))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # MLP无KV缓存，空实现

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # MLP无缓存依赖，空实现

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # MLP无缓存输出，空实现

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        """返回输入激活值的形状和数据类型（注意力层输出的隐藏状态）"""
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        """
        前向传播：执行MLP非线性变换
        Args:
            hidden: 隐藏状态容器（输入：注意力层输出；输出：MLP结果）
            weight_read_buf: 权重缓冲区
            i: 生成步骤索引
            k: GPU批处理索引
        """
        donate = [False] * 7  # 内存捐赠标记（优化GPU显存复用）
        h, donate[0] = hidden.val, True  # 输入隐藏状态标记为可复用

        # 读取权重：最后一批处理时释放缓冲区
        if k == self.policy.num_gpu_batches - 1:
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        # 调用GPU后端MLP计算（层归一化→升维→激活→降维→残差连接）
        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h  # 更新隐藏状态


class TransformerLayer:
    """
    Transformer层：组合自注意力模块（SelfAttention）和MLP模块
    实现流程：自注意力计算 → MLP非线性变换（均含层归一化和残差连接）
    统一管理权重、缓存和前向传播逻辑
    """
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)  # 自注意力模块
        self.mlp = MLP(config, env, policy, i)  # MLP模块
        self.policy = policy
        self.compute = self.attention.compute  # 继承注意力模块的计算设备

    def set_task(self, task):
        """绑定推理任务到子模块"""
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        """初始化子模块权重并统一存储"""
        attn_home, mlp_home = ValueHolder(), ValueHolder()
        self.attention.init_weight(attn_home, path)
        self.mlp.init_weight(mlp_home, path)
        weight_home.store((attn_home, mlp_home))

    def load_weight(self, weight_home, weight_read_buf, k):
        """加载子模块权重到缓冲区"""
        attn_read_buf, mlp_read_buf = ValueHolder(), ValueHolder()
        attn_home, mlp_home = weight_home.val
        self.attention.load_weight(attn_home, attn_read_buf, k)
        self.mlp.load_weight(mlp_home, mlp_read_buf, k)
        if k == 0:  # 仅首次批处理初始化缓冲区
            weight_read_buf.store((attn_read_buf, mlp_read_buf))

    def init_cache_one_gpu_batch(self, cache_home):
        """初始化KV缓存（仅注意力模块需要）"""
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, i):
        """加载KV缓存（仅注意力模块需要）"""
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        """存储KV缓存（仅注意力模块需要）"""
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        """
        前向传播：先执行自注意力，再执行MLP
        Args:
            cache_read_buf: 缓存读取缓冲区（供注意力模块使用）
            cache_write_buf: 缓存写入缓冲区（供注意力模块使用）
        """
        # 读取子模块权重缓冲区
        if k == self.policy.num_gpu_batches - 1:
            attn_read_buf, mlp_read_buf = weight_read_buf.pop()
        else:
            attn_read_buf, mlp_read_buf = weight_read_buf.val

        # 1. 自注意力计算（更新隐藏状态）
        self.attention.forward(hidden, cache_read_buf, attn_read_buf, attention_mask,
                               cache_write_buf, i, k)
        # 2. MLP计算（基于注意力输出更新隐藏状态）
        self.mlp.forward(hidden, None, mlp_read_buf, attention_mask, None, i, k)


class OptLM:
    """
    OPT模型统一封装类：整合输入嵌入、Transformer层、输出嵌入
    支持多设备资源分配、量化压缩、IO-计算重叠等推理优化
    """
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):
        # 解析模型配置
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        # 构建模型层序列
        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))  # 输入嵌入层
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                # 拆分模式：注意力和MLP作为独立层（支持并行调度）
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                # 整合模式：注意力+MLP封装为TransformerLayer（简化管理）
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))  # 输出嵌入层
        self.layers = layers
        self.num_layers = len(layers)

        # 配置激活值存储设备（仅支持单一设备，不支持混合）
        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError("激活值仅支持单一设备存储（GPU/CPU/磁盘）")

        # 初始化CUDA流（实现IO与计算并行）
        self.load_weight_stream = torch.cuda.Stream()  # 权重加载流
        self.load_cache_stream = torch.cuda.Stream()    # 缓存加载流
        self.store_cache_stream = torch.cuda.Stream()   # 缓存存储流

        # 初始化中间缓冲区（按层、GPU批处理维度分配）
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)  # 缓存存储容器
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)  # 缓存读取缓冲区
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)  # 缓存写入缓冲区
        self.weight_read_buf = array_1d(num_layers, ValueHolder)  # 权重读取缓冲区
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)  # 注意力掩码容器
        self.weight_home = array_1d(num_layers, ValueHolder)  # 权重存储容器

        self.task = None
        self.init_all_weights()  # 初始化所有层权重

    def set_task(self, task):
        """绑定推理任务（含prompt长度、生成长度、采样参数等）"""
        self.task = task
        for layer in self.layers:
            layer.set_task(task)

    def init_weight(self, j):
        """
        初始化第j层权重：自动处理权重路径扩展和缺失权重下载
        Args:
            j: 层索引（0=输入嵌入，-1=输出嵌入）
        """
        # 扩展权重路径（支持相对路径、家目录路径）
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        # 校验权重是否存在（以位置嵌入权重为校验基准）
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        # 权重缺失时自动下载（排除虚拟权重场景）
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)
        # 初始化当前层权重
        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        """
        加载权重到读取缓冲区（支持IO与计算重叠）
        Args:
            i: int，当前生成步骤索引
            j: int，当前层索引
            k: int，当前GPU批处理索引
            overlap: bool，是否启用IO与计算重叠（通过独立CUDA流实现）
        """
        # 边界处理：当层索引达到总层数时，切换到下一生成步骤，层索引重置为0
        if j == self.num_layers:
            j = 0
            i += 1
            # 若生成步骤已达到执行上限，直接返回（无需继续加载）
            if i == self.execute_gen_len:
                return

        # 从权重存储容器（weight_home）加载到权重读取缓冲区（weight_read_buf）
        if overlap:
            # 重叠模式：使用独立的权重加载CUDA流，不阻塞计算流
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            # 非重叠模式：同步加载，计算需等待IO完成
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        """
        删除指定层、指定GPU批次的权重数据，释放内存
        Args:
            j: int，层索引
            k: int，GPU批处理索引（仅k=0时执行，因权重仅首次加载复用）
        """
        if k == 0:
            # 弹出当前层的所有权重数据
            for x in self.weight_home[j].pop():
                # 处理嵌套的ValueHolder（如TransformerLayer的权重存储结构）
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()  # 递归删除底层权重数据
                else:
                    x.delete()  # 直接删除单一层权重数据

    def init_cache(self, j, k):
        """
        初始化指定层、指定GPU批处理的KV缓存
        Args:
            j: int，层索引
            k: int，GPU批处理索引
        """
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        """
        加载KV缓存到读取缓冲区（支持IO与计算重叠）
        Args:
            i: int，当前生成步骤索引
            j: int，当前层索引
            k: int，当前GPU批处理索引
            overlap: bool，是否启用IO与计算重叠
        """
        # 预填充阶段（i=0）无历史缓存，直接返回
        if i == 0:  # prefill, no cache
            return
        # 边界处理：批处理索引达到上限时，切换到下一层，批处理索引重置为0
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        # 边界处理：层索引达到上限时，切换到下一生成步骤，层索引重置为0
        if j == self.num_layers:
            j = 0
            i += 1
            # 若生成步骤已达到执行上限，直接返回
            if i == self.execute_gen_len:
                return

        # 从缓存存储容器（cache_home）加载到缓存读取缓冲区（cache_read_buf）
        if overlap:
            # 重叠模式：使用独立的缓存加载CUDA流，不阻塞计算流
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            # 非重叠模式：同步加载
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        """
        将KV缓存从写入缓冲区存储到存储容器（支持IO与计算重叠）
        Args:
            i: int，当前生成步骤索引
            j: int，当前层索引
            k: int，当前GPU批处理索引
            overlap: bool，是否启用IO与计算重叠
        """
        # 边界处理：批处理索引为-1时，切换到上一批次（最后一个批次），层索引减1
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        # 边界处理：层索引为-1时，切换到上一生成步骤，层索引设为最后一层
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            # 若生成步骤为-1（预填充前），直接返回
            if i == -1:
                return
        # 最后一个生成步骤无需存储缓存（无后续步骤复用），直接弹出缓冲区释放内存
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # 将缓存写入缓冲区（cache_write_buf）的数据存储到缓存存储容器（cache_home）
        if overlap:
            # 重叠模式：使用独立的缓存存储CUDA流，不阻塞计算流
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            # 非重叠模式：同步存储
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        """
        删除指定层、指定GPU批处理的KV缓存，释放内存
        Args:
            j: int，层索引
            k: int，GPU批处理索引
        """
        # 弹出缓存存储容器中的数据（KV缓存对）
        v = self.cache_home[j][k].pop()
        # 若缓存存在，递归删除K和V缓存数据
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        """
        加载隐藏状态到当前层输入缓冲区
        Args:
            i: int，当前生成步骤索引
            j: int，当前层索引
            k: int，当前GPU批处理索引
        """
        # 边界处理：批处理索引达到上限时，切换到下一层，批处理索引重置为0
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        # 边界处理：层索引达到上限时，切换到下一生成步骤，层索引重置为0
        if j == self.num_layers:
            j = 0
            i += 1
            # 若生成步骤已达到执行上限，直接返回
            if i == self.execute_gen_len:
                return

        # 获取当前层的计算设备
        dst = self.layers[j].compute
        if j == 0:  # 输入层：加载token ID作为输入
            gpu_batch_size = self.policy.gpu_batch_size
            # 计算当前GPU批次对应的样本范围（左闭右开）
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # 预填充阶段：加载整个prompt的token ID
                # 分配内存（形状：[gpu_batch_size, prompt_len]，数据类型：int32）
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                # 从输出容器中加载prompt部分的token ID
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # 生成阶段：加载上一步生成的最后一个token ID
                # 计算当前生成token在输出容器中的位置
                pos = self.task.prompt_len + i
                # 分配内存（形状：[gpu_batch_size, 1]，数据类型：int32）
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                # 加载上一步生成的token ID
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # 非输入层：加载上一层的输出作为当前层输入
            # 从上一层的隐藏状态缓冲区弹出数据，并移动到当前层计算设备
            val = self.hidden[i][j-1][k].pop().move(dst)
        # 将加载的隐藏状态存储到当前层输入缓冲区
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        """
        存储当前层的隐藏状态输出
        Args:
            i: int，当前生成步骤索引
            j: int，当前层索引
            k: int，当前GPU批处理索引
        """
        # 边界处理：批处理索引为-1时，切换到上一批次（最后一个批次），层索引减1
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        # 边界处理：层索引为-1时，切换到上一生成步骤，层索引设为最后一层
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            # 若生成步骤为-1（预填充前），直接返回
            if i == -1:
                return

        # 输出层：将生成的token ID存储到输出容器
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            # 计算当前GPU批次对应的样本范围（左闭右开）
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            # 从隐藏状态缓冲区弹出结果，转换为CPU端numpy数组
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            # 计算当前生成token在输出容器中的位置
            pos = self.task.prompt_len + i
            if self.task.stop:  # 启用stop token功能
                # 获取当前批次的样本停止标记
                stopped = self.stopped[left:right]
                # 已停止的样本填充pad token，未停止的样本存储生成结果
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                # 更新停止标记：生成stop token的样本标记为停止
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:  # 不启用stop token功能，直接存储生成结果
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # 非输出层：将隐藏状态移动到激活值存储设备（供下一层加载）
            x = self.hidden[i][j][k]
            # 避免重复移动（重叠模式下可能已完成移动）
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        """
        执行单层前向传播计算（原地更新隐藏状态）
        Args:
            i: int，当前生成步骤索引
            j: int，当前层索引
            k: int，当前GPU批处理索引
        """
        # 调用当前层的forward方法，完成层计算
        # 入参包含：隐藏状态（输入→输出原地更新）、缓存读写缓冲区、权重缓冲区、注意力掩码、生成步骤/批次索引
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k)

    def sync(self):
        """同步所有设备的操作，确保所有异步IO和计算完成"""
        self.env.disk.synchronize()  # 同步磁盘设备IO
        torch.cuda.synchronize()     # 同步所有CUDA流（计算、权重加载、缓存加载/存储）

    def init_all_weights(self):
        """初始化所有层的权重存储容器，并加载权重"""
        # 创建1维ValueHolder数组，存储各层权重
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        # 循环初始化每一层的权重
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        """删除所有层的权重数据，释放内存"""
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        """
        更新注意力掩码：预填充阶段生成padding掩码，生成阶段扩展掩码
        Args:
            i: int，当前生成步骤索引（i=0为预填充，i>0为生成）
            k: int，当前GPU批处理索引
        """
        if i > 0:  # 生成阶段：扩展掩码以包含当前生成的token
            mask = self.attention_mask[k]
            # 确保掩码已初始化
            assert mask.val is not None
            # 调用设备的掩码扩展方法，添加当前token的有效位置
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        # 预填充阶段：生成padding掩码（标记有效token，屏蔽pad token）
        gpu_batch_size = self.policy.gpu_batch_size
        # 计算当前GPU批次对应的样本范围（左闭右开）
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        # 获取当前批次的prompt token ID
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        # 确定注意力计算设备（CPU或GPU，由策略配置）
        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        # 分配掩码内存（形状：[gpu_batch_size, prompt_len]，数据类型：bool）
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        # 加载padding掩码：True表示有效token，False表示pad token
        val.load_from_np((input_ids != self.config.pad_token_id))
        # 存储到注意力掩码容器
        self.attention_mask[k].store(val)

    def generate(self,
                inputs: Union[np.array, List[List[int]]],
                max_new_tokens: int = 32,
                do_sample: bool = False,
                temperature: float = 1.0,
                stop: Optional[int] = None,
                debug_mode: Optional[str] = None,
                cut_gen_len: Optional[int] = None,
                verbose: int = 0):
        """
        模型推理主入口：接收输入prompt，生成指定长度的token序列
        Args:
            inputs: Union[np.array, List[List[int]]]，输入prompt的token ID（形状：[batch_size, prompt_len]）
            max_new_tokens: int，最大生成token数（默认32）
            do_sample: bool，是否启用随机采样（默认False，贪心解码）
            temperature: float，采样温度（仅do_sample=True时生效，默认1.0）
            stop: Optional[int]，stop token ID（生成到该token时停止，默认None）
            debug_mode: Optional[str]，调试模式（None/"fewer_batch"/"breakdown"，默认None）
            cut_gen_len: Optional[int]，截断生成长度（用于调试，默认None）
            verbose: int，日志输出级别（默认0，无额外输出）
        Returns:
            np.array: 生成的token ID序列（形状：[batch_size, prompt_len + max_new_tokens]）
        """
        # 创建推理任务实例，封装输入、生成参数等配置
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),  # prompt长度（假设所有样本prompt长度一致）
            gen_len=max_new_tokens,     # 最大生成长度
            cut_gen_len=cut_gen_len,    # 截断生成长度（调试用）
            do_sample=do_sample,        # 是否启用采样
            temperature=temperature,    # 采样温度
            stop=stop,                  # stop token ID
        )
        # 提取核心配置参数
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap  # 是否启用IO与计算重叠
        prompt_len, gen_len = task.prompt_len, task.gen_len
        # 确定实际执行的生成长度（优先使用截断长度，无则使用最大生成长度）
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # 初始化输出容器：存储prompt + 生成的token ID，默认用pad token填充
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        # 初始化样本停止标记：True表示已生成stop token，停止后续生成
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        # 将输入prompt的token ID填充到输出容器的前prompt_len列
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        # 校验批处理配置：总样本数必须等于GPU批处理数×单批样本数
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # 初始化中间缓冲区（清空历史数据）
        # 缓冲区维度说明：i（生成步骤）、j（层）、k（GPU批处理）
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        # 清空KV缓存相关缓冲区
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        # 清空权重读取缓冲区
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        # 清空注意力掩码缓冲区
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        # 初始化3D隐藏状态缓冲区：[gen_len, num_layers, num_gpu_batches]
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # 初始化缓存和任务配置
        self.set_task(task)  # 将任务绑定到所有层
        # 初始化所有层、所有GPU批次的KV缓存
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        # 若启用CPU缓存计算，初始化CPU端注意力计算工作空间
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # 根据模式选择生成循环逻辑
        if debug_mode is None:
            if not overlap:
                # 普通模式：无IO与计算重叠，逻辑清晰，适合调试
                self.generation_loop_normal()
            else:
                # 重叠模式：IO与计算并行，提升推理吞吐量
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()  # 单GPU批次重叠逻辑
                else:
                    self.generation_loop_overlap_multi_batch()   # 多GPU批次重叠逻辑
        elif debug_mode == "fewer_batch":
            # 调试模式：减少层和批次数量，快速验证逻辑
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # 调试模式：无重叠，拆分执行步骤，输出各环节耗时
            self.generation_loop_debug_normal()
        else:
            raise ValueError(f"无效的调试模式: {debug_mode}")

        # 生成完成后清理资源
        # 删除所有KV缓存
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        # 若启用CPU缓存计算，删除CPU端注意力计算工作空间
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        # 返回生成结果（包含原始prompt和生成的token ID）
        return self.output_ids

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

def generation_loop_debug_normal(self):
    """
    调试模式的普通生成循环（无IO与计算重叠）
    功能：拆分每个执行步骤，统计各环节耗时（权重加载、缓存加载/存储、层计算），支持限制执行批次
    """
    execute_num_batches = 20  # 限制调试时的执行批次数（避免耗时过长）
    batch_ct = 0  # 已执行的批次计数器
    pbar = tqdm(total=execute_num_batches)  # 进度条可视化

    # 初始化所有计时器（用于统计各环节耗时）
    timers("prefill_total").reset()  # 预填充阶段总耗时
    timers("decoding_gpu_batch").reset()  # 生成阶段单批次耗时
    timers("load_weight").reset()  # 权重加载耗时（跨阶段复用）
    timers("load_cache_prefill").reset()  # 预填充阶段缓存加载耗时
    timers("load_cache_decoding").reset()  # 生成阶段缓存加载耗时
    timers("store_cache_prefill").reset()  # 预填充阶段缓存存储耗时
    timers("store_cache_decoding").reset()  # 生成阶段缓存存储耗时
    timers("compute_layer_prefill").reset()  # 预填充阶段层计算耗时
    timers("compute_layer_decoding").reset()  # 生成阶段层计算耗时
    load_weight_timer = timers("load_weight")  # 权重加载计时器引用

    # 遍历所有生成步骤（i=0为预填充阶段，i>0为生成阶段）
    for i in range(self.execute_gen_len):
        if i == 0:
            # 预填充阶段：启动总计时器，绑定预填充阶段专属计时器
            timers("prefill_total").start()
            load_cache_timer = timers("load_cache_prefill")
            store_cache_timer = timers("store_cache_prefill")
            compute_layer_timer = timers("compute_layer_prefill")
        else:
            # 生成阶段：绑定生成阶段专属计时器
            load_cache_timer = timers("load_cache_decoding")
            store_cache_timer = timers("store_cache_decoding")
            compute_layer_timer = timers("compute_layer_decoding")

        # 更新所有GPU批次的注意力掩码（预填充生成padding掩码，生成阶段扩展掩码）
        for k in range(self.num_gpu_batches):
            self.update_attention_mask(i, k)

        # 遍历所有层，执行加载→计算→存储流程
        for j in range(self.num_layers):
            if i > 0:
                # 生成阶段：启动单批次耗时计时器
                timers("decoding_gpu_batch").start()

            # 1. 加载权重（跨批次复用，按层统计耗时）
            load_weight_timer.start(self.sync)  # 计时开始（同步确保前序操作完成）
            for k in range(self.num_gpu_batches):
                self.load_weight(i, j, k)
            load_weight_timer.stop(self.sync)  # 计时结束（同步确保加载完成）

            # 2. 遍历所有GPU批次，执行单批次完整流程
            for k in range(self.num_gpu_batches):
                # 加载缓存（按阶段统计耗时）
                load_cache_timer.start(self.sync)
                self.load_cache(i, j, k)
                load_cache_timer.stop(self.sync)

                # 加载隐藏状态（上一层输出或输入token ID）
                self.load_hidden(i, j, k)

                # 执行层计算（按阶段统计耗时）
                compute_layer_timer.start(self.sync)
                self.compute_layer(i, j, k)
                compute_layer_timer.stop(self.sync)

                # 存储隐藏状态（输出到下一层或最终结果）
                self.store_hidden(i, j, k)

                # 存储缓存（预填充/生成阶段的KV缓存）
                store_cache_timer.start(self.sync)
                self.store_cache(i, j, k)
                store_cache_timer.stop(self.sync)

            if i > 0:
                # 生成阶段：停止单批次计时器，更新进度条
                timers("decoding_gpu_batch").stop()
                pbar.update(1)
                batch_ct += 1
            # 达到限制批次数，提前退出循环
            if batch_ct >= execute_num_batches:
                break
        if batch_ct >= execute_num_batches:
            break
        if i == 0:
            # 预填充阶段结束，停止总计时器
            timers("prefill_total").stop(self.sync)

    # 转换计时器结果：将单批次生成耗时扩展为全量生成步骤耗时
    # 剔除前10个批次的热身数据，取平均单批次耗时
    batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
    for i in range(self.execute_gen_len):
        if i == 0:
            # 预填充阶段耗时直接复用
            timers("generate").costs.append(timers("prefill_total").costs[0])
        else:
            # 生成阶段：每层耗时×层数 = 单步骤总耗时
            timers("generate").costs.append(self.num_layers * batch_cost)

    # 输出调试信息：层数量、批次数、各环节平均耗时
    print(f"#layers: {self.num_layers}")
    print(f"#batches prefill:  {self.num_layers * self.num_gpu_batches}")
    print(f"#batches decoding: {(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
    print(f"load_weight            (per-layer): {np.mean(timers('load_weight').costs):.6f} s")
    # 输出预填充/生成阶段的各环节耗时
    for stage in ["prefill", "decoding"]:
        for func in ["load_cache", "store_cache", "compute_layer"]:
            name = func + "_" + stage
            costs = timers(name).costs
            print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

def generation_loop_overlap_single_batch(self):
    """
    单GPU批次的重叠生成循环（IO与计算并行）
    核心优化：提前加载下一层的权重和缓存，与当前层计算并行，提升吞吐量
    """
    # 前置操作（Prologue）：预加载第0层权重（启动阶段无并行空间）
    for k in range(self.num_gpu_batches):
        self.load_weight(0, 0, k)
    self.sync()  # 同步确保权重加载完成

    # 生成循环：逐步骤执行预填充和生成
    for i in range(self.execute_gen_len):
        timers("generate").start()  # 启动单步骤总计时器

        # 更新当前步骤的注意力掩码（单批次仅需处理k=0）
        self.update_attention_mask(i, 0)

        # 遍历所有层，执行并行加载→计算→存储
        for j in range(self.num_layers):
            # 提前加载下一层（j+1）的权重和缓存（与当前层计算并行）
            self.load_weight(i, j+1, 0)
            self.load_cache(i, j+1, 0)

            # 加载当前层的隐藏状态
            self.load_hidden(i, j, 0)

            # 执行当前层计算
            self.compute_layer(i, j, 0)

            # 存储上一层（j-1）的缓存（与当前层计算并行）
            self.store_cache(i, j-1, 0)

            # 存储当前层的隐藏状态
            self.store_hidden(i, j, 0)

            self.sync()  # 同步确保当前层所有操作完成

        timers("generate").stop()  # 停止单步骤计时器

        # 若所有样本均已生成stop token，提前退出循环
        if self.task.stop and np.all(self.stopped):
            break

def generation_loop_overlap_multi_batch(self):
    """
    多GPU批次的重叠生成循环（IO与计算并行+多批次流水线）
    核心优化：多批次流水线执行，同时并行加载下一批次/下一层的资源，最大化硬件利用率
    """
    # 前置操作（Prologue）：预加载第0层权重和第0批次的隐藏状态
    for k in range(self.num_gpu_batches):
        self.load_weight(0, 0, k)
    self.load_hidden(0, 0, 0)
    self.sync()  # 同步确保前置操作完成

    # 生成循环：逐步骤执行预填充和生成
    for i in range(self.execute_gen_len):
        timers("generate").start()  # 启动单步骤总计时器

        # 更新所有GPU批次的注意力掩码
        for k in range(self.num_gpu_batches):
            self.update_attention_mask(i, k)

        # 遍历所有层，多批次并行执行
        for j in range(self.num_layers):
            for k in range(self.num_gpu_batches):
                # 提前加载下一层（j+1）的权重 + 下一批次（k+1）的缓存
                self.load_weight(i, j+1, k)
                self.load_cache(i, j, k+1)

                # 存储上一批次（k-1）的隐藏状态和缓存
                self.store_hidden(i, j, k-1)
                self.store_cache(i, j, k-1)

                # 加载下一批次（k+1）的隐藏状态
                self.load_hidden(i, j, k+1)

                # 执行当前批次、当前层的计算
                self.compute_layer(i, j, k)

                self.sync()  # 同步确保当前批次操作完成

        timers("generate").stop()  # 停止单步骤计时器

    # 后置操作（Epilogue）：存储最后一个步骤、最后一层、最后一个批次的隐藏状态
    self.store_hidden(
        self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

def generation_loop_debug_single_batch(self):
    """
    单GPU批次的调试模式生成循环（简化重叠逻辑，限制执行批次）
    功能：快速验证单批次重叠逻辑，统计核心耗时，支持提前退出
    """
    execute_num_batches = 20  # 限制调试执行批次数
    batch_ct = 0  # 已执行批次计数器
    pbar = tqdm(total=execute_num_batches)  # 进度条

    # 初始化计时器
    timers("prefill").reset()  # 预填充阶段耗时
    timers("decoding_gpu_batch").reset()  # 生成阶段单批次耗时

    # 前置操作：预加载第0层权重
    for k in range(self.num_gpu_batches):
        self.load_weight(0, 0, k)
    self.sync()

    # 生成循环
    for i in range(self.execute_gen_len):
        if i == 0:
            timers("prefill").start()  # 预填充阶段启动计时器
        # 更新注意力掩码（单批次k=0）
        self.update_attention_mask(i, 0)

        # 遍历所有层
        for j in range(self.num_layers):
            if i > 0:
                timers("decoding_gpu_batch").start()  # 生成阶段启动单批次计时器

            # 提前加载下一层权重和缓存
            self.load_weight(i, j+1, 0)
            self.load_cache(i, j+1, 0)

            # 加载当前层隐藏状态
            self.load_hidden(i, j, 0)

            # 执行层计算
            self.compute_layer(i, j, 0)

            # 存储上一层缓存
            self.store_cache(i, j-1, 0)

            # 存储当前层隐藏状态
            self.store_hidden(i, j, 0)

            self.sync()  # 同步确保当前层操作完成

            if i > 0:
                # 生成阶段：停止计时器，更新进度条
                timers("decoding_gpu_batch").stop()
                pbar.update(1)
                batch_ct += 1
            # 达到限制批次数，提前退出
            if batch_ct >= execute_num_batches:
                break
        if batch_ct >= execute_num_batches:
            break
        if i == 0:
            timers("prefill").stop()  # 预填充阶段停止计时器

    # 转换计时器结果：将单批次耗时扩展为全量生成步骤耗时
    batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])  # 剔除热身数据
    for i in range(self.execute_gen_len):
        if i == 0:
            timers("generate").costs.append(timers("prefill").costs[0])
        else:
            timers("generate").costs.append(self.num_layers * batch_cost)

def generation_loop_debug_multi_batch(self):
    """
    多GPU批次的调试模式生成循环（简化重叠逻辑，限制执行批次）
    功能：快速验证多批次重叠逻辑，统计核心耗时，支持提前退出
    """
    execute_num_batches = 20  # 限制调试执行批次数
    batch_ct = 0  # 已执行批次计数器
    pbar = tqdm(total=execute_num_batches)  # 进度条

    # 初始化计时器
    timers("prefill").reset()  # 预填充阶段耗时
    timers("decoding_gpu_batch").reset()  # 生成阶段单批次耗时

    # 前置操作：预加载第0层权重和第0批次隐藏状态
    for k in range(self.num_gpu_batches):
        self.load_weight(0, 0, k)
    self.load_hidden(0, 0, 0)
    self.sync()

    # 生成循环
    for i in range(self.execute_gen_len):
        if i == 0:
            timers("prefill").start()  # 预填充阶段启动计时器
        # 更新所有GPU批次的注意力掩码
        for k in range(self.num_gpu_batches):
            self.update_attention_mask(i, k)

        # 遍历所有层
        for j in range(self.num_layers):
            if i > 0:
                timers("decoding_gpu_batch").start()  # 生成阶段启动单批次计时器

            # 遍历所有GPU批次，执行简化的重叠逻辑
            for k in range(self.num_gpu_batches):
                # 提前加载下一层权重和下一批次缓存
                self.load_weight(i, j+1, k)
                self.load_cache(i, j, k+1)

                # 存储上一批次隐藏状态和缓存
                self.store_hidden(i, j, k-1)
                self.store_cache(i, j, k-1)

                # 加载下一批次隐藏状态
                self.load_hidden(i, j, k+1)

                # 执行当前层计算
                self.compute_layer(i, j, k)

                self.sync()  # 同步确保当前批次操作完成

            if i > 0:
                # 生成阶段：停止计时器，更新进度条
                timers("decoding_gpu_batch").stop()
                pbar.update(1)
                batch_ct += 1
            # 达到限制批次数，提前退出
            if batch_ct >= execute_num_batches:
                break
        if batch_ct >= execute_num_batches:
            break
        if i == 0:
            timers("prefill").stop()  # 预填充阶段停止计时器

    # 转换计时器结果：将单批次耗时扩展为全量生成步骤耗时
    batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])  # 剔除热身数据
    for i in range(self.execute_gen_len):
        if i == 0:
            timers("generate").costs.append(timers("prefill").costs[0])
        else:
            timers("generate").costs.append(self.num_layers * batch_cost)

def __del__(self):
    """析构函数：对象销毁时自动删除所有权重数据，释放内存资源"""
    self.delete_all_weights()


def get_filename(args):
    """
    根据命令行参数生成日志文件名（包含核心配置，便于区分实验结果）
    Args:
        args: 命令行参数对象
    Returns:
        str: 生成的日志文件名
    """
    # 提取模型大小（如opt-30b的"30b"）
    model_size = args.model.split('-')[-1]
    # 拼接量化比例字符串（如[100,0,100,0,100,0]→"100-0-100-0-100-0-"）
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    # 构建基础文件名（包含模型大小、批处理配置、prompt长度、生成长度、量化比例）
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    # 补充缓存计算设备标识（CPU/GPU）
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    # 补充权重压缩标识
    if args.compress_weight:
        filename += "-compw"
    # 补充缓存压缩标识
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    """
    生成测试用输入token ID（固定prompt重复多次，适配批处理）
    Args:
        prompt_len: int，prompt的固定长度（不足则padding）
        num_prompts: int，生成的prompt数量（等于总样本数）
        tokenizer: 分词器对象
    Returns:
        tuple: 测试输入token ID元组（shape: [num_prompts, prompt_len]）
    """
    # 固定测试prompt（"Paris is the capital city of"）
    prompts = ["Paris is the capital city of"]
    # 分词并padding到固定长度
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    # 重复prompt至指定数量（适配批处理）
    return (input_ids[0],) * num_prompts


def run_flexllmgen(args):
    """
    FlexLLMGen推理主函数：初始化环境、模型、策略，执行热身和基准测试，输出日志
    Args:
        args: 命令行参数对象（包含模型配置、批处理参数、量化压缩等）
    """
    print(f"<run_flexllmgen>: args.model: {args.model}")
    # 初始化分词器（根据模型选择对应tokenizer，padding方向设为左）
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    # 计算总样本数（GPU批处理数 × 单批样本数）
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # 生成测试输入（热身用+基准测试用）
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)  # 热身用短prompt
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)  # 基准测试用指定长度prompt

    # 初始化执行环境（GPU/CPU/磁盘设备）
    gpu = TorchDevice("cuda:0")  # GPU设备（默认cuda:0）
    cpu = TorchDevice("cpu")     # CPU设备
    disk = TorchDisk(args.offload_dir)  # 磁盘卸载目录
    # 混合设备（整合GPU/CPU/磁盘，用于跨设备数据调度）
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    # 初始化推理策略（批处理、量化压缩、缓存配置等）
    policy = Policy(
        gpu_batch_size=args.gpu_batch_size,  # 单GPU批次样本数
        num_gpu_batches=args.num_gpu_batches,  # GPU批处理数
        weight_gpu_percent=args.percent[0],  # 权重GPU存储比例
        weight_cpu_percent=args.percent[1],  # 权重CPU存储比例
        cache_gpu_percent=args.percent[2],   # 缓存GPU存储比例
        cache_cpu_percent=args.percent[3],   # 缓存CPU存储比例
        act_gpu_percent=args.percent[4],     # 激活值GPU存储比例
        act_cpu_percent=args.percent[5],     # 激活值CPU存储比例
        overlap=args.overlap,  # 是否启用IO与计算重叠
        sep_layer=args.sep_layer,  # 是否拆分Transformer层（注意力+MLP独立）
        pin_weight=args.pin_weight,  # 是否固定权重内存（避免分页）
        cpu_cache_compute=args.cpu_cache_compute,  # 是否在CPU上计算注意力缓存
        attn_sparsity=args.attn_sparsity,  # 注意力稀疏度（1.0为稠密）
        compress_weight=args.compress_weight,  # 是否压缩权重
        compress_weight_config=CompressionConfig(  # 权重压缩配置（4bit量化）
            num_bits=4, group_size=64, group_dim=0, symmetric=False
        ),
        compress_cache=args.compress_cache,  # 是否压缩缓存
        compress_cache_config=CompressionConfig(  # 缓存压缩配置（4bit量化）
            num_bits=4, group_size=64, group_dim=2, symmetric=False
        )
    )
    # 断言：缓存压缩与注意力稀疏度不兼容（未实现）
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    # 加载模型配置，计算资源占用（模型大小、缓存大小、激活值大小）
    opt_config = get_opt_config(args.model)
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)  # 总缓存大小（字节）
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)  # 总激活值大小（字节）
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    # 初始化模型（加载权重）
    print("init weight...")
    model = OptLM(opt_config, env, args.path, policy)

    try:
        # 热身推理（避免首次执行的初始化开销影响基准测试）
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        # 基准测试推理
        print("benchmark - generate")
        timers("generate").reset()  # 重置生成计时器
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs  # 获取各步骤耗时
    finally:
        # 关闭环境的复制线程（释放资源）
        env.close_copy_threads()

    # 计算性能指标（延迟、吞吐量）
    prefill_latency = costs[0]  # 预填充阶段延迟（秒）
    prefill_throughput = num_prompts * prompt_len / prefill_latency  # 预填充吞吐量（token/秒）
    if cut_gen_len:
        # 若截断生成长度，将测试结果投影到完整生成长度
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        # 完整生成长度：生成阶段总延迟（所有步骤耗时之和）
        decode_latency = sum(costs[1:])
    # 生成阶段吞吐量（token/秒）
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    total_latency = prefill_latency + decode_latency  # 总延迟（秒）
    num_generated_tokens = num_prompts * gen_len  # 总生成token数
    total_throughput = num_generated_tokens / total_latency  # 总吞吐量（token/秒）
    # 获取设备峰值内存占用
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    # 解码生成结果并输出（非虚拟权重场景）
    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        # 输出第1个和最后1个样本的结果（简化输出）
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    # 输出设备资源统计（GPU/CPU内存使用）
    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)  # 是否为投影后的性能指标

    # 生成日志文件名（自动或指定）
    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    # 写入基准测试日志
    log_str = write_benchmark_log(
        filename=filename,
        model_size=opt_config.model_bytes(),
        cache_size=cache_size,
        hidden_size=hidden_size,
        gpu_peak_mem=gpu_peak_mem,
        projected=projected,
        prefill_latency=prefill_latency,
        prefill_throughput=prefill_throughput,
        decode_latency=decode_latency,
        decode_throughput=decode_throughput,
        total_latency=total_latency,
        total_throughput=total_throughput
    )
    # 输出日志内容（verbose级别≥1时）
    if args.verbose >= 1:
        print(log_str)


def add_parser_arguments(parser):
    """
    向命令行解析器添加FlexLLMGen的配置参数
    Args:
        parser: argparse.ArgumentParser对象
    """
    # 模型配置
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="模型名称（如facebook/opt-6.7b、facebook/galactica-30b）")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="模型权重路径。若路径中无缓存权重，FlexLLMGen会自动从HuggingFace下载")
    parser.add_argument("--offload-dir", type=str, default="~/flexllmgen_offload_dir",
        help="张量卸载目录（磁盘存储路径，用于存放CPU/GPU放不下的权重/缓存）")

    # 序列长度配置
    parser.add_argument("--prompt-len", type=int, default=512,
        help="输入prompt的固定长度（不足则padding，过长则截断）")
    parser.add_argument("--gen-len", type=int, default=32,
        help="最大生成token数")
    parser.add_argument("--cut-gen-len", type=int,
        help="调试用截断生成长度（快速验证逻辑，无需完整生成）")
    parser.add_argument("--debug-mode", type=str, choices=["fewer_batch", "breakdown"],
        help="调试模式：fewer_batch（减少批次/层数）、breakdown（拆分步骤耗时）")

    # 批处理配置
    parser.add_argument("--gpu-batch-size", type=int, default=4,
        help="单GPU批次的样本数")
    parser.add_argument("--num-gpu-batches", type=int, default=1,
        help="GPU批处理数（总样本数=num_gpu_batches × gpu_batch_size）")

    # 设备存储比例配置（6个数值分别对应：权重GPU/CPU、缓存GPU/CPU、激活值GPU/CPU）
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="6个整数，分别表示：权重GPU占比、权重CPU占比、缓存GPU占比、缓存CPU占比、激活值GPU占比、激活值CPU占比")

    # 模型结构与内存配置
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True,
        help="是否拆分Transformer层（将注意力和MLP作为独立层，支持并行调度）")
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True,
        help="是否固定权重内存（避免操作系统内存分页，提升访问速度）")
    parser.add_argument("--cpu-cache-compute", action="store_true",
        help="是否在CPU上计算注意力缓存（适用于GPU内存不足场景）")
    parser.add_argument("--attn-sparsity", type=float, default=1.0,
        help="注意力稀疏度（1.0为稠密注意力，<1.0为稀疏注意力，需配合稀疏实现）")

    # 量化压缩配置
    parser.add_argument("--compress-weight", action="store_true",
        help="是否启用权重压缩（默认4bit量化，配置见CompressionConfig）")
    parser.add_argument("--compress-cache", action="store_true",
        help="是否启用缓存压缩（默认4bit量化，配置见CompressionConfig）")

    # 日志与输出配置
    parser.add_argument("--log-file", type=str, default="auto",
        help="日志文件名。设为auto时自动根据参数生成文件名")
    parser.add_argument("--no-log", action="store_true",
        help="是否禁用日志写入（仅输出到控制台）")
    parser.add_argument("--verbose", type=int, default=2,
        help="日志输出级别：0（无输出）、1（仅性能指标）、2（性能指标+生成结果）")

    # 优化策略配置
    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True,
        help="是否启用IO与计算重叠（通过CUDA流并行加载资源，提升吞吐量）")


if __name__ == "__main__":
    """主函数：解析命令行参数，执行FlexLLMGen推理流程"""
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)  # 添加配置参数
    args = parser.parse_args()  # 解析参数

    # 校验参数：percent必须包含6个数值
    assert len(args.percent) == 6, "参数--percent必须包含6个整数（权重GPU/CPU、缓存GPU/CPU、激活值GPU/CPU占比）"

    # 执行推理流程
    run_flexllmgen(args)
