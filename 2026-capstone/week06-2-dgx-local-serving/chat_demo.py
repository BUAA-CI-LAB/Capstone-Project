"""调用本地 vLLM 服务（OpenAI 兼容接口），流式打印回答，并统计 TTFT 和生成速度。

用法：
    python chat_demo.py --port 8103
    python chat_demo.py --port 8103 --prompt "解释一下 KV cache" --thinking
"""

import argparse
import time

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="调用本地 vLLM 服务并统计速度")
    parser.add_argument("--port", type=int, required=True, help="vllm serve 的端口：8100 + 组号")
    parser.add_argument("--model", help="请求的模型名，默认取服务上的第一个模型")
    parser.add_argument("--prompt", default="用三句话介绍一下 vLLM。")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true", help="打开 Qwen3 的思考模式（默认关闭）")
    parser.add_argument("--api-key", default="EMPTY", help="服务端加了 --api-key 时填同一个值")
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key=args.api_key)
    model = args.model or client.models.list().data[0].id

    start = time.perf_counter()
    first_token_at = None
    usage = None
    section = None
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": args.thinking}},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # 服务端加了 --reasoning-parser 时，思考内容单独放在 reasoning 字段
        for name, text in (("思考", getattr(delta, "reasoning", None)), ("回答", delta.content)):
            if not text:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            if section != name:
                print(f"\n[{name}]")
                section = name
            print(text, end="", flush=True)
    end = time.perf_counter()

    print("\n")
    if first_token_at is None or usage is None:
        print("没有收到输出，检查端口和服务日志。")
        return
    decode_tokens = usage.completion_tokens - 1
    decode_seconds = end - first_token_at
    speed = decode_tokens / decode_seconds if decode_seconds > 0 else float("nan")
    print(f"模型 {model} | 输入 {usage.prompt_tokens} token | 输出 {usage.completion_tokens} token")
    print(f"TTFT {first_token_at - start:.3f} s | 生成速度 {speed:.1f} token/s | 总耗时 {end - start:.2f} s")


if __name__ == "__main__":
    main()
