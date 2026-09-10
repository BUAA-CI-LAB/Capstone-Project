import argparse
import torch
import time  # 用于计时
from transformers import AutoTokenizer
from flexllmgen.flex_opt import OptLM, Policy, ExecutionEnv
from flexllmgen.opt_config import get_opt_config
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice

def main():
    # ========== 外部参数接口定义（所有可调整参数） ==========
    parser = argparse.ArgumentParser(description="FlexGen参数化推理脚本（支持OPT模型）")
    
    # 1. 模型与硬件配置
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
                        help="Hugging Face模型ID（如facebook/opt-1.3b）")
    parser.add_argument("--gpu-memory", type=int, default=6,
                        help="GPU显存分配上限（GB）")
    parser.add_argument("--cpu-memory", type=int, default=16,
                        help="CPU内存分配上限（GB）")
    parser.add_argument("--percent", nargs="+", type=int, 
                        default=[10, 90, 100, 0, 100, 0],
                        help="6个数字：[权重GPU%, 权重CPU%, 缓存GPU%, 缓存CPU%, 激活值GPU%, 激活值CPU%]")
    
    # 2. 批量处理参数
    parser.add_argument("--gpu-batch-size", type=int, default=2,
                        help="单GPU的批处理大小（原hardcode的2）")
    parser.add_argument("--num-gpu-batches", type=int, default=1,
                        help="GPU上的总批次数（原hardcode的1，单卡场景通常为1）")
    
    # 3. 输入与生成配置
    parser.add_argument("--prompt", type=str, default="what is AI",
                        help="输入的提示文本（原hardcode的'what is AI'）")
    parser.add_argument("--max-input-length", type=int, default=32,
                        help="输入文本的最大长度（超过截断，原hardcode的32）")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="生成的新token数量（原hardcode的32）")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度（0-1，越高多样性越强，原hardcode的0.7）")
    
    # 解析参数
    args = parser.parse_args()

    # ========== 2. 初始化设备（CPU+GPU协同） ==========
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk("./offload_dir")
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    # ========== 3. 定义Offloading策略（使用外部参数） ==========
    policy = Policy(
        gpu_batch_size=args.gpu_batch_size,          # 外部参数：单GPU批大小
        num_gpu_batches=args.num_gpu_batches,        # 外部参数：总批次数
        w_gpu_percent=args.percent[0],
        w_cpu_percent=args.percent[1],
        cache_gpu_percent=args.percent[2],
        cache_cpu_percent=args.percent[3],
        act_gpu_percent=args.percent[4],
        act_cpu_percent=args.percent[5],
        overlap=True,
        sep_layer=True,
        pin_weight=True,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=False,
        comp_weight_config=None,
        compress_cache=False,
        comp_cache_config=None
    )

    # 打印当前参数配置（方便实验记录）
    print("===== 当前运行参数 =====")
    print(f"模型：{args.model} | GPU显存：{args.gpu_memory}GB | CPU内存：{args.cpu_memory}GB")
    print(f"Offloading策略：权重[{args.percent[0]}%GPU, {args.percent[1]}%CPU] | "
          f"缓存[{args.percent[2]}%GPU, {args.percent[3]}%CPU] | "
          f"激活值[{args.percent[4]}%GPU, {args.percent[5]}%CPU]")
    print(f"批量配置：单GPU批大小={args.gpu_batch_size} | 总批次数={args.num_gpu_batches}")
    print(f"生成配置：输入长度={args.max_input_length} | 新token数={args.max_new_tokens} | 温度={args.temperature}")
    print(f"输入prompt：{args.prompt}")
    print("=========================\n")

    # ========== 4. 加载模型配置和分词器 ==========
    opt_config = get_opt_config(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 修复OPT模型无pad_token的问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"已自动设置pad_token为：{tokenizer.pad_token}")

    # ========== 5. 初始化模型（计时） ==========
    print(f"从镜像站下载模型 {args.model}（CPU+GPU协同）...")
    start_load = time.time()  # 记录加载开始时间
    model = OptLM(
        config=opt_config,
        env=env,
        path="~/.cache/huggingface/hub",
        policy=policy
    )
    load_time = time.time() - start_load  # 模型加载时间
    print(f"模型加载完成，耗时：{load_time:.2f}秒")

    # ========== 6. 准备输入（使用外部prompt参数） ==========
    inputs = tokenizer(
        args.prompt,  # 外部参数：输入文本
        padding="max_length",
        max_length=args.max_input_length,  # 外部参数：输入长度
        truncation=True,
        return_tensors="pt"
    ).input_ids.numpy()
    # 计算总样本数（批大小×批次数）
    total_samples = args.gpu_batch_size * args.num_gpu_batches
    inputs = [inputs[0]] * total_samples  # 适配总样本数
    print(f"\n输入文本处理完成，总样本数：{total_samples}")

    # ========== 7. 推理（计时+统计吞吐） ==========
    print("开始推理...")
    start_infer = time.time()  # 记录推理开始时间
    output_ids = model.generate(
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,  # 外部参数：新token数
        do_sample=True,
        temperature=args.temperature,        # 外部参数：温度
    )
    infer_time = time.time() - start_infer  # 推理总时间

    # 计算吞吐率（总生成token数 / 推理时间）
    total_generated_tokens = total_samples * args.max_new_tokens  # 总生成token数
    throughput = total_generated_tokens / infer_time  # 吞吐率（tokens/秒）

    # ========== 8. 输出结果 ==========
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("\n===== 生成结果 =====")
    for i, out in enumerate(outputs):
        print(f"样本 {i+1}:\n{out}\n")

    # 输出时间和吞吐统计
    print("===== 性能指标 =====")
    print(f"模型加载时间：{load_time:.2f}秒")
    print(f"推理总时间：{infer_time:.2f}秒")
    print(f"总生成token数：{total_generated_tokens}")
    print(f"吞吐率：{throughput:.2f} tokens/秒")  # 每秒生成的token数
    

if __name__ == "__main__":
    main()