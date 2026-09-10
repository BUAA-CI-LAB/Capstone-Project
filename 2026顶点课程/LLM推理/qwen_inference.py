"""单条 prompt 的 Qwen 推理、计时与张量观察。

模型下载方式（先按 README.md 安装依赖）：

推荐使用 ModelScope 下载（在终端依次执行）：
   python -m pip install modelscope
   python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct')"
   python qwen_inference.py --model ./models/Qwen2.5-1.5B-Instruct --device cpu --debug

也可使用 Hugging Face：

1. 自动下载：首次使用模型 ID 运行时，from_pretrained 会从 Hugging Face
   下载 tokenizer 和权重到缓存；后续运行复用缓存。
   python qwen_inference.py --model Qwen/Qwen2.5-1.5B-Instruct --device cpu --debug

2. 提前下载到指定目录（在终端执行以下一行）：
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct')"
   然后从本地目录加载：
   python qwen_inference.py --model ./models/Qwen2.5-1.5B-Instruct --device cpu --debug

3B 模型将上述模型 ID 和目录中的 1.5B 替换为 3B 即可。
下载需要能访问所选模型站点；也可在联网机器上下载后，将完整目录复制到实验机器。
目录需包含 config、tokenizer 和全部权重文件。下载时间不计入 generation_seconds。
"""
import argparse
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_summary(tensor, sample):
    # 只拷贝一个位置的少量数值，避免打印或保存整份大张量。
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sample": sample.detach().float().cpu().tolist(),
    }


@torch.inference_mode()
def inspect_tensors(model, inputs):
    """额外执行 prefill 和最多一步 decode，不计入正式推理耗时。"""
    out = model(**inputs, use_cache=True, output_hidden_states=True,
                return_dict=True)
    info = {"hidden_states_count": len(out.hidden_states), "hidden_states": {}}
    for index in sorted({0, 1, len(out.hidden_states) - 1}):
        hidden = out.hidden_states[index]
        info["hidden_states"][str(index)] = tensor_summary(hidden, hidden[0, -1, :8])
    info["logits"] = tensor_summary(out.logits, out.logits[0, -1, :8])
    scores, ids = out.logits[0, -1].topk(5)
    info["top5_token_ids"] = ids.cpu().tolist()
    info["top5_logits"] = scores.float().cpu().tolist()
    cache = out.past_key_values
    key, value = cache[0]
    info["prefill_cache_layer0"] = {
        "key": tensor_summary(key, key[0, 0, -1, :8]),
        "value": tensor_summary(value, value[0, 0, -1, :8]),
    }
    next_id = out.logits[:, -1:].argmax(-1)
    info["first_generated_token_id"] = next_id.item()
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else ([] if eos is None else [eos])
    del out, key, value, hidden
    if next_id.item() in eos_ids:
        info["decode_skipped"] = "Prefill 已预测 EOS，无需继续 decode。"
        return info
    mask = torch.cat((inputs.attention_mask, inputs.attention_mask.new_ones((1, 1))), -1)
    position = torch.tensor([inputs.input_ids.shape[1]], device=next_id.device)
    decoded = model(input_ids=next_id, attention_mask=mask,
                    position_ids=position.unsqueeze(0), cache_position=position,
                    past_key_values=cache, use_cache=True,
                    output_hidden_states=True, return_dict=True)
    info["decode_input_shape"] = list(next_id.shape)
    info["decode_attention_mask_shape"] = list(mask.shape)
    info["decode_last_hidden"] = tensor_summary(
        decoded.hidden_states[-1], decoded.hidden_states[-1][0, -1, :8])
    info["decode_cache_key_shape"] = list(decoded.past_key_values[0][0].shape)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Hugging Face 模型 ID 或本地完整模型目录")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt", default="中国的首都是哪里？请用一句话回答。")
    parser.add_argument("--max-new-tokens", type=positive_int, default=32)
    parser.add_argument("--debug", action="store_true", help="额外观察 hidden states 和 KV cache")
    parser.add_argument("--output", type=Path, default=Path("results/run.json"))
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("prompt 不能为空")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请安装匹配的 PyTorch 或改用 --device cpu")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    dtype = (torch.float32 if device.type == "cpu" else
             torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    print(f"Loading {args.model}; device={device}, dtype={dtype}", flush=True)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    synchronize(device)
    load_seconds = time.perf_counter() - start
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    # 用单独的配置固定 greedy 策略，避免继承模型的采样参数。
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        # 一次完整预热，再对一次完整 generate 计时；不包含下载、分词、打印。
        print("Warmup...", flush=True)
        warmup_ids = model.generate(**inputs, generation_config=config)
        synchronize(device)
        del warmup_ids
        start = time.perf_counter()
        generated = model.generate(**inputs, generation_config=config)
        synchronize(device)
        elapsed = time.perf_counter() - start
    new_ids = generated[0, inputs.input_ids.shape[1]:].cpu().tolist()
    result = {
        "model": args.model, "device": str(device), "dtype": str(dtype),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda"
        else platform.processor() or platform.machine(),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(), "prompt": args.prompt,
        "input_ids": inputs.input_ids[0].cpu().tolist(),
        "input_shape": list(inputs.input_ids.shape),
        "attention_mask_shape": list(inputs.attention_mask.shape),
        "input_tokens": inputs.input_ids.shape[1],
        "max_new_tokens": args.max_new_tokens, "generated_tokens": len(new_ids),
        "generated_token_ids": new_ids,
        "output_text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "load_seconds": load_seconds, "generation_seconds": elapsed,
        "tokens_per_second": len(new_ids) / elapsed,
        "warmup_runs": 1, "timed_runs": 1,
        "generation_settings": {"do_sample": False, "use_cache": True,
                                "attention": "eager"},
        "model_config": {name: getattr(model.config, name) for name in
                         ["num_hidden_layers", "hidden_size", "vocab_size",
                          "num_attention_heads", "num_key_value_heads"]},
    }
    if args.debug:
        result["debug"] = inspect_tensors(model, inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
