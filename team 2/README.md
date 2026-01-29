# EdgeLLM-Optimization

**Official repository for the deployment and optimization of autoregressive large language models on edge devices.**

This project demonstrates a full-stack solution for deploying high-performance Large Language Models (LLMs, e.g., Llama-3-8B) on consumer-grade mobile devices (e.g., Xiaomi 14 with Snapdragon 8 Gen 3). It covers the entire pipeline from **LoRA fine-tuning** and **Heterogeneous Quantization** (Q4_K_M weights + Q4_0 KV Cache) to **Android App deployment**.


## 🚀 Key Features

* **Efficient Fine-Tuning**: LoRA-based style transfer (Modern Chinese Poetry) using LLaMA-Factory.
* **Heterogeneous Quantization**:
* **Weights**: GGUF Q4_K_M (Mixed precision for critical layers).
* **KV Cache**: Block-wise Q4_0 dynamic quantization (70%+ memory reduction).


* **Edge Optimization**: ARM NEON optimized inference engine via `llama.cpp`.
* **Android Integration**: Custom JNI bindings and Android App for offline inference.

---

## 🛠️ Environment Setup

### 1. For Training & Quantization (PC/Server)

* **OS**: Linux (Ubuntu 20.04+) or WSL2
* **GPU**: NVIDIA GPU with CUDA support (for LoRA fine-tuning)
* **Dependencies**:
```bash
# Install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
pip install -r requirements.txt

# Install llama.cpp build tools
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

```



### 2. For Inference & Benchmarking (Android/Termux)

* **Device**: Android Device (Snapdragon 8 Gen 2/3 recommended, 12GB+ RAM)
* **Termux Environment**:
```bash
pkg install clang cmake git build-essential python

```



---

## 🎨 Fine-Tuning & Model Preparation

### Step 1: LoRA Fine-Tuning

We use `LLaMA-Factory` to fine-tune the base model (e.g., Llama-3-8B) on the poetry dataset.

**Example Command:**

```bash
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset chinese_modern_poetry \
    --template llama3 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir saves/llama3-poetry-lora \
    --fp16

```

### Step 2: Merge & Export to GGUF

Merge the LoRA adapter with the base model and convert to GGUF format.

```bash
# Merge LoRA and export to FP16 GGUF
python llama.cpp/convert-hf-to-gguf.py \
    path/to/base_model \
    --lora-path saves/llama3-poetry-lora \
    --outtype f16 \
    --outfile models/llama3-poetry-fp16.gguf

```

### Step 3: Quantization (Q4_K_M)

Quantize the model weights to 4-bit mixed precision.

```bash
./llama.cpp/llama-quantize \
    models/llama3-poetry-fp16.gguf \
    models/llama3-poetry-q4_k_m.gguf \
    Q4_K_M

```

---

## 📂 Model Weights Placement

For the Android App and Benchmarking scripts to function correctly, please organize your files as follows:

**1. For Android App Development:**
Place the `.gguf` model file in the Android project assets folder (or push to device storage if loading dynamically).

* Path: `android-app/app/src/main/assets/llama3-8b-q4_k_m.gguf`

**2. For Termux Benchmarking:**
Push the model to the device storage:

```bash
adb push models/llama3-poetry-q4_k_m.gguf /data/local/tmp/
# OR
adb push models/llama3-poetry-q4_k_m.gguf /sdcard/Download/

```

---

## 📱 Android App Configuration

This project contains a complete Android Studio project in the `android-app/` directory.

1. **Prerequisites**: Android Studio Hedgehog or later, NDK (Side-by-side) 26.x.
2. **Open Project**: Open the `android-app` folder in Android Studio.
3. **Configure JNI**:
* The project uses `CMakeLists.txt` to build `libllama.so` locally.
* Ensure `local.properties` points to your NDK installation.


4. **Build & Run**:
* Connect your Xiaomi 14 via USB debugging.
* Run the `app` configuration.


---

## 📊 KV Cache Benchmarking

We provide a script `benchmark_kv.py` to evaluate the impact of KV Cache quantization on memory usage and inference speed.

### Usage

1. **Copy Benchmark Tools to Device**:
Ensure `llama-cli` (compiled for ARM) and `benchmark_kv.py` are on the device.
2. **Run the Benchmark**:
```bash
python3 benchmark_kv.py \
    --model /data/local/tmp/llama3-poetry-q4_k_m.gguf \
    --prompt-file prompts/long_context.txt \
    --ctx-size 4096 \
    --kv-type q4_0  # Options: f16, q8_0, q4_0

```



### Expected Output

The script will parse the `llama_print_timings` and memory logs to output:

* **KV Cache Memory Usage**: (e.g., ~63 MiB for Q4_0 vs ~224 MiB for FP16)
* **Prompt Processing Speed**: (tokens/sec)
* **Generation Speed**: (tokens/sec)

---

## 🤝 Acknowledgements

* [llama.cpp](https://github.com/ggerganov/llama.cpp) for the core inference engine.
* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the efficient fine-tuning framework.
* [Termux](https://termux.dev/) for the on-device Linux environment.

## 📄 License

[MIT License](https://www.google.com/search?q=LICENSE)