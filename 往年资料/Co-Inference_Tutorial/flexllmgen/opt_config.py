"""
OPT模型配置类与权重下载工具
用于定义不同规模OPT模型的参数配置，并提供从Hugging Face下载预训练权重
并转换为numpy格式的功能

部分函数改编自：https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model
"""

import argparse
import dataclasses
import glob
import os
import shutil

import numpy as np
from tqdm import tqdm  # 进度条显示工具


@dataclasses.dataclass(frozen=True)  # 不可变数据类（配置参数一旦定义不可修改）
class OptConfig:
    """
    OPT模型配置类：存储不同规模OPT模型的结构参数和属性
    支持计算模型权重、缓存、隐藏层的存储占用
    """
    name: str = "opt-125m"  # 模型名称（默认125M规模）
    num_hidden_layers: int = 12  # Transformer隐藏层数量
    max_seq_len: int = 2048  # 最大序列长度
    hidden_size: int = 768  # 隐藏层维度（与input_dim一致）
    n_head: int = 12  # 注意力头数
    input_dim: int = 768  # 输入/输出维度（模型核心维度）
    ffn_embed_dim: int = 3072  # MLP层中间维度（默认hidden_size×4）
    pad: int = 1  # 填充标记ID（兼容旧版本配置）
    activation_fn: str = 'relu'  # 激活函数（OPT默认ReLU）
    vocab_size: int = 50272  # 词汇表大小（GPT-2兼容）
    layer_norm_eps: float = 0.00001  # 层归一化epsilon（数值稳定性）
    pad_token_id: int = 1  # 填充token ID（与pad一致）
    dtype: type = np.float16  # 模型数据类型（默认FP16，节省存储）

    def model_bytes(self):
        """
        计算模型权重总占用字节数（仅权重，不含缓存/中间结果）
        计算公式基于OPT模型结构：嵌入层 + 各隐藏层（自注意力+MLP+层归一化）
        Returns:
            模型权重总字节数（int）
        """
        h = self.input_dim  # 核心维度（hidden_size）
        return 2 * (self.num_hidden_layers * (
            # 自注意力层权重：QKV投影（3h×h + 3h偏置） + 输出投影（h×h + h偏置）
            h * (3 * h + 1) + h * (h + 1) +
            # MLP层权重：中间投影（4h×h + 4h偏置） + 输出投影（h×4h + h偏置）
            h * (4 * h + 1) + h * 4 * (h + 1) +
            # 层归一化权重：每个隐藏层4个LN（自注意力输入/输出、MLP输入/输出）
            h * 4) +
            # 嵌入层权重：token嵌入（vocab_size×h） + 位置嵌入（max_seq_len×h）
            self.vocab_size * (h + 1))

    def cache_bytes(self, batch_size, seq_len):
        """
        计算注意力缓存（K/V）总占用字节数
        Args:
            batch_size: 推理批大小
            seq_len: 序列长度（输入+生成）
        Returns:
            缓存总字节数（int）
        """
        # 2（K+V）× 批大小 × 序列长度 × 层数 × 核心维度 × 2（FP16占2字节）
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        """
        计算隐藏层输出总占用字节数（中间激活值）
        Args:
            batch_size: 推理批大小
            seq_len: 序列长度
        Returns:
            隐藏层总字节数（int）
        """
        # 批大小 × 序列长度 × 核心维度 × 2（FP16占2字节）
        return batch_size * seq_len * self.input_dim * 2


def get_opt_config(name, **kwargs):
    """
    根据模型名称获取对应的OptConfig配置（支持自定义参数覆盖）
    Args:
        name: 模型名称（如"opt-125m", "opt-13b", "galactica-30b"）
        **kwargs: 自定义参数（用于覆盖默认配置）
    Returns:
        初始化后的OptConfig实例
    """
    # 处理带路径的模型名称（如"facebook/opt-125m" → "opt-125m"）
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()  # 转为小写，兼容大小写输入

    # 处理IML变体模型（opt-iml-30b → 按opt-30b配置）
    if "-iml-max" in name:
        arch_name = name.replace("-iml-max", "")
    elif "-iml" in name:
        arch_name = name.replace("-iml", "")
    else:
        arch_name = name

    # 不同规模模型的配置参数（核心差异：层数、头数、核心维度）
    if arch_name == "opt-125m":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=12,
            n_head=12,
            hidden_size=768,
            input_dim=768,
            ffn_embed_dim=768 * 4,  # MLP中间维度=核心维度×4
        )
    elif arch_name == "opt-350m":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=24,
            n_head=16,
            hidden_size=1024,
            input_dim=1024,
            ffn_embed_dim=1024 * 4,
        )
        raise NotImplementedError("该模型架构未实现（结构与标准OPT不同）")
    elif arch_name == "opt-1.3b":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=24,
            n_head=32,
            hidden_size=2048,
            input_dim=2048,
            ffn_embed_dim=2048 * 4,
        )
    elif arch_name == "opt-2.7b":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=32,
            n_head=32,
            hidden_size=2560,
            input_dim=2560,
            ffn_embed_dim=2560 * 4,
        )
    elif arch_name == "opt-6.7b":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=32,
            n_head=32,
            hidden_size=4096,
            input_dim=4096,
            ffn_embed_dim=4096 * 4,
        )
    elif arch_name == "opt-13b":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=40,
            n_head=40,
            hidden_size=5120,
            input_dim=5120,
            ffn_embed_dim=5120 * 4,
        )
    elif arch_name == "opt-30b":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=48,
            n_head=56,
            hidden_size=7168,
            input_dim=7168,
            ffn_embed_dim=7168 * 4,
        )
    elif arch_name == "galactica-30b":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=48,
            n_head=56,
            hidden_size=7168,
            input_dim=7168,
            ffn_embed_dim=7168 * 4,
            vocab_size=50000,  # Galactica词汇表大小与OPT不同
        )
    elif arch_name == "opt-66b":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=64,
            n_head=72,
            hidden_size=9216,
            input_dim=9216,
            ffn_embed_dim=9216 * 4,
        )
    elif arch_name == "opt-175b":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=96,
            n_head=96,
            hidden_size=12288,
            input_dim=12288,
            ffn_embed_dim=12288 * 4,
        )
    elif arch_name == "opt-175b-stage":
        config = OptConfig(
            name=name,
            max_seq_len=2048,
            num_hidden_layers=24,  # 简化版175B（仅24层）
            n_head=96,
            hidden_size=12288,
            input_dim=12288,
            ffn_embed_dim=12288 * 4,
        )
    else:
        raise ValueError(f"无效的模型名称: {name}")

    # 用自定义参数覆盖默认配置（如修改dtype、max_seq_len等）
    return dataclasses.replace(config, **kwargs)


def download_opt_weights_old(model_name, path):
    """
    旧版权重下载函数：从Hugging Face下载PyTorch权重并转换为numpy格式
    支持OPT、Bloom、Galactica系列模型
    Args:
        model_name: 模型名称（如"opt-125m", "bloom-560m"）
        path: 权重保存根路径（最终保存到{path}/{model_name}-np）
    """
    import torch
    from transformers import OPTForCausalLM, BloomForCausalLM  # 导入Hugging Face模型类

    # 处理带路径的模型名称
    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    # 构建最终保存路径（numpy格式权重目录）
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))  # 解析用户目录（~→/home/user）

    # 映射模型名称到Hugging Face仓库名称和模型类
    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
        model_class = OPTForCausalLM
    elif "bloom" in model_name:
        hf_model_name = "bigscience/" + model_name
        model_class = BloomForCausalLM
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name
        model_class = OPTForCausalLM  # Galactica基于OPT架构
    else:
        raise ValueError(f"无效的模型名称: {model_name}")

    print(f"从Hugging Face加载{model_name}的PyTorch预训练权重...")
    print(f"下载和CPU加载可能需要数十分钟（取决于网络和模型大小）。")
    print(f"若长时间无响应，可通过监控进程内存占用确认进度。")

    disable_torch_init()  # 禁用PyTorch冗余初始化（加速模型加载）
    # 加载预训练模型（FP16精度，快速初始化）
    model = model_class.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float16,
        _fast_init=True
    )
    restore_torch_init()  # 恢复PyTorch默认初始化

    # 创建保存目录（若不存在）
    os.makedirs(path, exist_ok=True)

    print(f"将权重转换为numpy格式并保存到 {path} ...")
    if "opt" in model_name or "galactica" in model_name:
        # OPT/Galactica模型：遍历decoder层参数
        for name, param in tqdm(list(model.model.named_parameters())):
            # 统一层归一化名称（final_layer_norm → layer_norm）
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)  # 每个参数单独保存为.npy文件
            with open(param_path, "wb") as f:
                # PyTorch张量→CPU→numpy→保存
                np.save(f, param.cpu().detach().numpy())
    elif "bloom" in model_name:
        # Bloom模型：遍历transformer层参数
        for name, param in tqdm(list(model.transformer.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    else:
        raise ValueError(f"无效的模型名称: {model_name}")


# 全局变量：保存PyTorch默认初始化函数（用于后续恢复）
global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    禁用PyTorch层的默认初始化（加速模型加载）
    原理：预训练模型加载时无需重新初始化权重，禁用后跳过冗余步骤
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    # 保存Linear和LayerNorm的默认初始化函数
    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters

    # 替换为空函数（禁用初始化）
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """恢复PyTorch层的默认初始化函数（禁用后需调用）"""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def disable_hf_opt_init():
    """禁用Hugging Face OPT模型的冗余初始化（加速加载）"""
    import transformers
    # 替换OPT预训练模型的_init_weights方法为空函数
    setattr(transformers.models.opt.modeling_opt.OPTPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)


def download_opt_weights(model_name, path):
    """
    新版权重下载函数：通过snapshot_download下载.bin权重文件，再转换为numpy
    支持OPT和Galactica模型，下载更稳定（直接获取权重文件）
    Args:
        model_name: 模型名称（如"opt-125m", "galactica-30b"）
        path: 权重保存根路径
    """
    from huggingface_hub import snapshot_download  # 从Hugging Face Hub下载快照
    import torch

    print(f"从Hugging Face加载{model_name}的PyTorch预训练权重...")
    print(f"下载和CPU加载可能需要数十分钟（取决于网络和模型大小）。")
    print(f"若长时间无响应，可通过监控进程内存占用确认进度。")

    # 映射模型名称到Hugging Face仓库名称
    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name
    else:
        raise ValueError(f"仅支持OPT和Galactica模型，当前输入: {model_name}")

    # 下载模型快照（仅获取.bin权重文件）
    folder = snapshot_download(hf_model_name, allow_patterns="*.bin")
    bin_files = glob.glob(os.path.join(folder, "*.bin"))  # 所有权重文件路径

    # 处理模型名称和保存路径
    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)  # 创建保存目录

    # 遍历所有.bin权重文件，转换为numpy格式
    for bin_file in tqdm(bin_files, desc="转换权重格式"):
        # 加载.bin文件中的权重字典（key: 参数名称，value: PyTorch张量）
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            # 清理参数名称（移除"model."前缀）
            name = name.replace("model.", "")
            # 统一层归一化名称（final_layer_norm → layer_norm）
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # 共享嵌入层权重：lm_head.weight与decoder.embed_tokens.weight相同
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(
                    param_path,
                    param_path.replace("decoder.embed_tokens.weight", "lm_head.weight")
                )


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="OPT模型权重下载工具")
    parser.add_argument("--model", type=str, required=True, help="模型名称（如opt-125m, opt-13b）")
    parser.add_argument("--path", type=str, default="~/opt_weights", help="权重保存根路径（默认~/opt_weights）")
    args = parser.parse_args()

    # 执行权重下载和转换
    download_opt_weights(args.model, args.path)