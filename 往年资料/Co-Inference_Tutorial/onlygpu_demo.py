import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    # 与FlexGen版本保持一致的参数接口
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--prompt", type=str, default="what is AI", help="输入提示文本")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="生成的新token数量")
    parser.add_argument("--gpu-batch-size", type=int, default=2, help="单GPU批大小")
    args = parser.parse_args()

    # 检查GPU是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("纯GPU推理需要CUDA环境，请确保GPU可用")
    device = torch.device("cuda:0")
    print(f"使用GPU设备：{device}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 修复pad_token

    # 加载模型（纯GPU，全部参数放GPU）
    print("\n===== 开始加载模型（纯GPU） =====")
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,  # 用半精度节省显存
        device_map="auto"  # 自动将模型加载到GPU
    ).to(device)
    model.eval()  # 推理模式
    load_time = time.time() - start_load
    print(f"模型加载完成，耗时：{load_time:.2f}秒")

    # 准备输入（批量处理）
    inputs = tokenizer(
        [args.prompt] * args.gpu_batch_size,  # 复制prompt生成批量输入
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt"
    ).to(device)  # 输入放GPU
    total_samples = args.gpu_batch_size
    print(f"\n输入文本：{args.prompt} | 总样本数：{total_samples}")

    # 推理（计时）
    print("\n===== 开始推理（纯GPU） =====")
    start_infer = time.time()
    with torch.no_grad():  # 关闭梯度计算，节省显存
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )
    infer_time = time.time() - start_infer  # 推理耗时

    # 计算吞吐（总生成token数 / 推理时间）
    total_generated_tokens = total_samples * args.max_new_tokens  # 总生成token数
    throughput = total_generated_tokens / infer_time  # 吞吐率（tokens/秒）

    # 输出结果
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("\n===== 生成结果 =====")
    for i, out in enumerate(outputs):
        print(f"样本 {i+1}:\n{out}\n")

    # 输出时间和吞吐统计
    print("===== 性能指标 =====")
    print(f"模型加载时间：{load_time:.2f}秒")
    print(f"推理时间（总）：{infer_time:.2f}秒")
    print(f"总生成token数：{total_generated_tokens}")
    print(f"吞吐率：{throughput:.2f} tokens/秒")

if __name__ == "__main__":
    main()