import transformers
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_model import llama_attn_forward_StreamingLLM
from llama_model import llama_sdpa_attn_forward_StreamingLLM
from llama_model import prepare_inputs_for_generation_llama, prepare_inputs_for_generation_llama_new

def replace_llama(method, model_name=None):
    if method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_StreamingLLM
        
    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new

def benchmark(model, tokenizer, prompt, max_new_tokens=128, device="cuda:1"):
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    # warmup （不计时）
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8)
    torch.cuda.synchronize()

    # 正式计时
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time
    output_length = output.shape[1]
    tokens_generated = output_length - input_length
    throughput = tokens_generated / elapsed

    # decode 输出
    new_tokens = output[0, input_length:]
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "elapsed": elapsed,
        "throughput": throughput,
        "input_length": input_length,
        "output_length": tokens_generated,
        "tokens_generated": tokens_generated,
        "decoded_output": decoded_output,
    }

def load_model(model_name, attn_impl="sdpa", device="cuda:1"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={ "": device },
        attn_implementation=attn_impl,
    )
    return model

def run_and_log(name, model, tokenizer, prompt, device):
    print(f"\n=== {name} ===")
    result = benchmark(model, tokenizer, prompt, device=device)

    print(f"输入长度: {result['input_length']} tokens")
    print(f"输出长度: {result['output_length']} tokens")
    print(f"耗时: {result['elapsed']:.2f}s")
    print(f"吞吐量: {result['throughput']:.2f} tokens/s")
    print(f"输出:\n{result['decoded_output']}\n")
    return result

def main():
    model_name = "/mtc/longlingkun/models/llama3.1-8b-instruct"
    device = "cuda:2"

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # === 加载长 prompt 文件 ===
    with open("long_prompt.txt", "r", encoding="utf-8") as f:
        long_prompt = f.read()
        
    # long_prompt = "I like large language models"

    # Baseline: 原始 SPDA
    model = load_model(model_name, attn_impl="sdpa", device=device)
    run_and_log("Baseline (SPDA Attention)", model, tokenizer, long_prompt, device)

    # StreamingLLM + SPDA
    # replace_llama("streamingllm")
    # model = load_model(model_name, attn_impl="sdpa", device=device)
    # run_and_log("StreamingLLM + SPDA Attention", model, tokenizer, long_prompt, device)

if __name__ == "__main__":
    main()