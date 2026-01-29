import subprocess
import re
import os
import sys

# === 配置区域 ===
BINARY_PATH = "build/bin/llama-cli"
MODEL_PATH = "models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
PROMPT_FILE = "prompt.txt"

# 上下文长度
CONTEXT_SIZE = 4096
# 预测长度
PREDICT_LEN = 100 

# === 检查环境 ===
if not os.path.exists(BINARY_PATH):
    print(f"❌ 错误: 找不到可执行文件: {BINARY_PATH}")
    sys.exit(1)
if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误: 找不到模型文件: {MODEL_PATH}")
    sys.exit(1)
if not os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, "w") as f:
        f.write("This is a test sentence to simulate long context. " * 300)
    print(f"   已生成 {PROMPT_FILE}")

# === 定义测试组 ===
kv_configs = [
    ("f16", "f16", "Baseline (FP16)"),
    ("q8_0", "q8_0", "Quantized (Q8_0)"),
    ("q4_0", "q4_0", "Aggressive (Q4_0)")
]

results = []

print(f"\n🚀 开始基准测试 (已修正 Regex 匹配)")
print(f"📂 模型: {os.path.basename(MODEL_PATH)}")
print("-" * 60)

for k_type, v_type, label in kv_configs:
    print(f"\n正在测试: {label} [K={k_type} / V={v_type}] ...")
    
    cmd = [
        f"./{BINARY_PATH}",
        "-m", MODEL_PATH,
        "-f", PROMPT_FILE,
        "-c", str(CONTEXT_SIZE),
        "-n", str(PREDICT_LEN),
        "--temp", "0",
        "--cache-type-k", k_type,
        "--cache-type-v", v_type,
        "--simple-io",
        "-no-cnv" 
    ]
    
    try:
        # timeout 设置为 600 秒防止卡死
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        output = result.stdout + result.stderr
        
        # --- 🔍 修正后的正则匹配 ---
        
        # 1. 提取 KV Cache 显存占用
        # 你的日志格式: "llama_kv_cache: size = 63.00 MiB"
        # 兼容旧格式: "kv self size = ..."
        kv_match = re.search(r'(?:llama_kv_cache:\s+size|kv self size)\s+=\s+([\d\.]+)\s+MiB', output)
        if kv_match:
            kv_size = float(kv_match.group(1))
        else:
            kv_size = 0.0
            print("   ⚠️ 警告: 未抓取到显存数据，可能日志格式又有变化。")

        # 2. 提取推理速度
        # 你的日志格式: "... ( 23.83 ms per token, 41.97 tokens per second)"
        # 策略: 找到 "tokens per second" 前面的最后一个浮点数
        speed_match = re.search(r'([\d\.]+)\s+tokens per second', output)
        if speed_match:
            speed = float(speed_match.group(1))
        else:
            speed = 0.0
            print("   ⚠️ 警告: 未抓取到速度数据。")
            
        # --- 🔍 Debug 信息 (如果全为0则打印日志片段) ---
        if kv_size == 0.0 and speed == 0.0:
            print("\n   [DEBUG] 抓取失败，原始日志片段:")
            print(output[-1000:]) # 打印最后1000字符帮助排查
            print("   [DEBUG] 结束\n")

        print(f"   ✅ 结果: 显存占用 = {kv_size} MiB | 速度 = {speed} T/s")
        
        results.append({
            "Label": label,
            "Memory_MiB": kv_size,
            "Speed_TPS": speed
        })
        
    except subprocess.TimeoutExpired:
        print("   ❌ 错误: 运行超时！")
    except Exception as e:
        print(f"   ❌ 运行出错: {e}")

# === 输出表格 ===
print("\n" + "="*60)
print(f"{'配置 (Configuration)':<25} | {'KV显存 (MiB)':<15} | {'速度 (Tokens/s)':<15}")
print("-" * 60)
for row in results:
    print(f"{row['Label']:<25} | {row['Memory_MiB']:<15} | {row['Speed_TPS']:<15}")
print("="*60)