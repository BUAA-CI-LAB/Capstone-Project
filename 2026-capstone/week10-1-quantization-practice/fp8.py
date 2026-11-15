#!/usr/bin/env python3
"""把 BF16 模型量化成 FP8（W8A8-FP8，权重 per-channel 静态、激活值 per-token 动态）。

和第九周课件的 FP8 Dynamic 是同一个方案，不需要校准数据。
用法（在单独的量化环境里运行，见 README「选做：自己量化」）：
    python fp8.py --src /srv/spark/models/Qwen3-8B --dst ~/models/Qwen3-8B-FP8-self
依赖：llmcompressor
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="BF16 模型目录或仓库名")
    ap.add_argument("--dst", required=True, help="量化产物输出目录")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype="auto", device_map="auto")
    tok = AutoTokenizer.from_pretrained(args.src)

    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)

    model.save_pretrained(args.dst, save_compressed=True)
    tok.save_pretrained(args.dst)
    elapsed = time.time() - t0

    # 产物检查：体积、是否含 quantization_config、文件列表
    cfg = json.load(open(os.path.join(args.dst, "config.json"), encoding="utf-8"))
    qc = cfg.get("quantization_config")
    files = sorted(os.listdir(args.dst))
    size_gb = sum(os.path.getsize(os.path.join(args.dst, f)) for f in files) / 1e9
    report = {
        "src": args.src, "dst": args.dst, "elapsed_s": round(elapsed, 1),
        "size_gb": round(size_gb, 2), "files": files,
        "has_quantization_config": qc is not None,
        "quantization_config": qc,
    }
    with open(os.path.join(args.dst, "quantize_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("elapsed_s", "size_gb", "has_quantization_config")},
                     ensure_ascii=False))
    if qc is None:
        raise SystemExit("config.json 里没有 quantization_config，产物不可用")


if __name__ == "__main__":
    main()
