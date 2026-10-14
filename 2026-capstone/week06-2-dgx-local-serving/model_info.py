"""读取本地模型的 config.json，打印结构参数，估算权重和 KV cache 占用。

用法：
    python model_info.py /srv/spark/models/Qwen3-1.7B
    python model_info.py /srv/spark/models/Qwen3-1.7B --kv-gib 4.5   # 4.5 换成 vLLM 日志里的 KV cache 内存
"""

import argparse
import json
from pathlib import Path

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def main() -> None:
    parser = argparse.ArgumentParser(description="估算模型权重和 KV cache 占用")
    parser.add_argument("model_dir", type=Path, help="模型目录，例如 /srv/spark/models/Qwen3-1.7B")
    parser.add_argument("--kv-gib", type=float, help="vLLM 启动日志中 Available KV cache memory 的数值（GiB）")
    args = parser.parse_args()

    cfg = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    layers = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    dtype = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    nbytes = DTYPE_BYTES.get(dtype, 2)
    max_len = cfg.get("max_position_embeddings")
    tied = cfg.get("tie_word_embeddings", False)

    for key, value in [
        ("model_type", cfg.get("model_type")),
        ("num_hidden_layers", layers),
        ("hidden_size", cfg["hidden_size"]),
        ("num_attention_heads", heads),
        ("num_key_value_heads", kv_heads),
        ("head_dim", head_dim),
        ("vocab_size", cfg["vocab_size"]),
        ("tie_word_embeddings", tied),
        ("max_position_embeddings", max_len),
        ("torch_dtype", dtype),
    ]:
        print(f"{key:<26}{value}")

    weight_bytes = sum(f.stat().st_size for f in args.model_dir.glob("*.safetensors"))
    kv_per_token = 2 * layers * kv_heads * head_dim * nbytes  # 每层 K、V 各一份

    print()
    print(f"权重文件合计：{weight_bytes / 2**30:.2f} GiB")
    index = args.model_dir / "model.safetensors.index.json"
    if tied and index.exists() and "lm_head.weight" in json.loads(index.read_text(encoding="utf-8"))["weight_map"]:
        shared = cfg["vocab_size"] * cfg["hidden_size"] * nbytes
        print(f"  其中 lm_head 与 embedding 共用的 {shared / 2**30:.2f} GiB 在文件里存了两份，加载后约 {(weight_bytes - shared) / 2**30:.2f} GiB")
    print(f"每个 token 的 KV cache：2 × {layers} 层 × {kv_heads} 个 KV 头 × {head_dim} 维 × {nbytes} 字节 = {kv_per_token / 1024:.0f} KiB")
    if max_len:
        print(f"一个 {max_len} token 的请求需要 KV cache：{kv_per_token * max_len / 2**30:.2f} GiB")
    if args.kv_gib:
        print(f"{args.kv_gib} GiB 的 KV cache 约能存 {int(args.kv_gib * 2**30 / kv_per_token):,} 个 token")


if __name__ == "__main__":
    main()
