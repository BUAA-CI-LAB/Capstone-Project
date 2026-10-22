"""测量本地 vLLM 服务的 TTFT、TPOT、ITL、端到端时延与吞吐（仅使用标准库）。

在 DGX Spark 上启动 vLLM 后运行，例如：
    python bench_lab.py --base-url http://127.0.0.1:8103/v1 --model qwen3-1.7b \
        --concurrency 4 --num-prompts 16 --max-tokens 128 --label c4 --output results/c4.json

重复长前缀（用于 Prefix Caching 实验）：
    python bench_lab.py --base-url http://127.0.0.1:8103/v1 --model qwen3-1.7b \
        --concurrency 1 --num-prompts 10 --shared-prefix-words 400 --label prefix-on

指标口径：
    TTFT  = 首个输出 token 到达时间 - 请求发送时间
    TPOT  = (端到端时延 - TTFT) / (输出 token 数 - 1)
    ITL   = 相邻两个输出 token 到达时间之差
    吞吐  = 全部请求输出 token 总数 / 全部请求完成所用墙钟时间
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[idx]


def summarize(values):
    return {
        "mean": statistics.mean(values) if values else float("nan"),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def one_request(base_url, api_key, payload):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    start = time.perf_counter()
    token_times = []
    usage = {}
    try:
        with urllib.request.urlopen(req) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning"):
                    token_times.append(time.perf_counter())
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    end = time.perf_counter()
    completion = int(usage.get("completion_tokens") or len(token_times))
    if not token_times or completion <= 0:
        return {"ok": False, "error": "no output tokens"}
    ttft = token_times[0] - start
    e2e = end - start
    itls = [b - a for a, b in zip(token_times, token_times[1:])]
    return {
        "ok": True,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "tpot_s": (e2e - ttft) / (completion - 1) if completion > 1 else float("nan"),
        "itl_mean_s": statistics.mean(itls) if itls else float("nan"),
        "itl_p99_s": percentile(itls, 99) if itls else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="测量 vLLM 服务的 TTFT / TPOT / 吞吐")
    parser.add_argument("--base-url", required=True, help="例如 http://127.0.0.1:8103/v1")
    parser.add_argument("--model", required=True, help="与 --served-model-name 一致")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=1, help="同时在途的请求数")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="用三句话介绍 vLLM 的 Continuous Batching。")
    parser.add_argument("--shared-prefix-words", type=int, default=0,
                        help="每个请求前重复的共享前缀词数，用于 Prefix Caching 实验")
    parser.add_argument("--label", default="run", help="本组测量名称，如 c1、c4、prefix-on")
    parser.add_argument("--output", type=Path, help="结果 JSON 路径，如 results/c4.json")
    args = parser.parse_args()

    prefix = ""
    if args.shared_prefix_words > 0:
        prefix = "课程实验共享背景材料：vLLM 使用连续组批与分页 KV Cache 管理请求。 " * (args.shared_prefix_words // 12 + 1)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prefix + args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(lambda _: one_request(args.base_url, args.api_key, payload), range(args.num_prompts)))
    wall = time.perf_counter() - wall_start

    ok = [r for r in records if r["ok"]]
    total_out = sum(r["completion_tokens"] for r in ok)
    summary = {
        "label": args.label,
        "requests_ok": len(ok),
        "requests_failed": len(records) - len(ok),
        "wall_s": round(wall, 3),
        "throughput_req_per_s": len(ok) / wall if wall else float("nan"),
        "throughput_output_tok_per_s": total_out / wall if wall else float("nan"),
        "ttft_s": summarize([r["ttft_s"] for r in ok]),
        "tpot_s": summarize([r["tpot_s"] for r in ok if r["tpot_s"] == r["tpot_s"]]),
        "e2e_s": summarize([r["e2e_s"] for r in ok]),
    }
    print(f"[{args.label}] ok={summary['requests_ok']} failed={summary['requests_failed']} "
          f"wall={summary['wall_s']}s req/s={summary['throughput_req_per_s']:.2f} "
          f"output tok/s={summary['throughput_output_tok_per_s']:.1f}")
    for name in ("ttft_s", "tpot_s", "e2e_s"):
        s = summary[name]
        print(f"  {name:<7} mean={s['mean']*1000:.1f}ms p50={s['p50']*1000:.1f}ms "
              f"p95={s['p95']*1000:.1f}ms p99={s['p99']*1000:.1f}ms")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cfg = {k: str(v) for k, v in vars(args).items()}
        args.output.write_text(json.dumps({"config": cfg, "summary": summary, "requests": records},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
