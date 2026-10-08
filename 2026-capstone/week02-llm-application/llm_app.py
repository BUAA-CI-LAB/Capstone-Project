"""第二周：单文件最小 LLM 应用。沿用第一周环境，运行后自动启动并关闭本地 API。

python llm_app.py --model models/Qwen2.5-1.5B-Instruct --device cuda --stream --example --chat
"""
import argparse
import ast
import asyncio
import json
import platform
import queue
import socket
import sys
import threading
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError


# 调用关系：main 加载模型和启动服务 → conversation 发出 HTTP 请求 →
# create_api 中的处理函数执行 generate → conversation 接收正文 → 校验并保存。
# 单轮调用结束即退出；聊天模式复用同一个模型，并在每次请求中携带问答历史。

# 1. Prompt 与输出约定。
# USER_PROMPT 是本轮任务；SYSTEM_PROMPT 定义角色；FORMAT_PROMPT 约定输出格式。
# 格式要求只通过 Prompt 提出，模型仍可能违反约定，因此后面必须用程序校验。
# 不带 --structured / --example 时只使用角色与任务，返回值按普通文本处理。
USER_PROMPT = "写一个统计列表中正数个数的 Python 函数。"
SYSTEM_PROMPT = "你是一个 Python 代码助手。"
FORMAT_PROMPT = (
    "只返回一个 JSON 对象，恰好包含 function_name 和 code 两个字符串字段。"
    "回答从 { 开始，到 } 结束；不要 Markdown 代码围栏或其他文字。"
    "code 中的换行用 JSON 的\\n转义。注意 Python 关键字和变量之间的空格。"
)
# --example 选择下面的参考请求，并自动开启结构化校验；未开启时使用 --prompt
# 或 USER_PROMPT。--prompt 与 --example 互斥，避免第一轮请求来源不明确。
# 这里用一个输入/输出示例说明格式；r 字符串让示例中的 \n 保持为两个字面字符。
# r 只影响 Python 源码中的字符串解释；响应仍需按 JSON 规则解析和还原转义。
EXAMPLE_PROMPT = r'''按照以下示例的形式回答，示例只用于展示格式：
请求：生成判断整数是否为偶数的函数 is_even(n)。
回答：
{"function_name":"is_even","code":"def is_even(n):\n    return n % 2 == 0"}

现在处理本次请求：生成一个简短的 Python 函数 count_positive(nums)。
nums 是一个数值列表，返回其中严格大于 0 的元素个数。
零和负数不计入；空列表返回 0。无需导入任何库。
遍历列表时，循环变量使用 x，比较条件是 x > 0。

只返回一个合法的 JSON 对象，恰好包含两个必填字段：
1. function_name：字符串，值为 count_positive。
2. code：字符串，包含完整函数定义，尽量使用两行代码。
代码字符串中的换行必须用 JSON 的 \n 转义，字符串用双引号。
不要 Markdown 代码围栏，不要解释，不要增加其他字段。
回答的第一个字符必须是 {，最后一个字符必须是 }。直接输出 JSON：
'''


# CodeResult 定义程序能够接受的输出结构；它校验模型正文，不是 HTTP 请求。
class CodeResult(BaseModel):
    # strict 禁止把数字等类型自动转成字符串；extra="forbid" 禁止约定以外的字段。
    # 下面两个字段没有默认值，因此都是必填项；这些规则也会导出到 JSON Schema。
    model_config = ConfigDict(strict=True, extra="forbid")
    function_name: str
    # str 只约束类型，默认允许空串；function_name 也没有限定为某个具体名称。
    # 拓展约束时可使用 Field(min_length=1) 等规则，并同步调整 FORMAT_PROMPT。
    code: str


def validate_text(text):
    # 输入是模型返回的完整正文字符串；输出字典记录校验状态、错误或解析结果。
    # 三层检查各有含义：JSON 能否解析 → 字段是否合规 → code 是否符合 Python 语法。
    # 即使全部通过，也不等于算法正确；功能正确性仍需另外检查。
    try:
        data = json.loads(text)  # 保留原文，不删围栏、不补齐 JSON。
    except json.JSONDecodeError as exc:
        return {"status": "json_error", "error": str(exc)}
    try:
        result = CodeResult.model_validate(data)
    except ValidationError as exc:
        return {"status": "schema_error", "error": str(exc)}
    # 只有字段约定通过后才能读取 result.code。ast.parse 构建语法树，可以发现
    # 缩进和括号等问题，但不会检查返回值是否符合题意，也不保证存在指定函数。
    try:
        ast.parse(result.code)  # 只解析语法，不执行模型生成的代码。
        syntax = {"status": "syntax_valid"}
    except (SyntaxError, ValueError, UnicodeError) as exc:
        syntax = {"status": "syntax_error", "error": str(exc), "line": getattr(exc, "lineno", None)}
    # Schema 与 Python 语法分开记录：schema_valid 可以同时伴随 syntax_error。
    # model_dump 得到普通字典；与导出规则的 model_json_schema() 用途不同。
    return {"status": "schema_valid", "parsed": result.model_dump(),
            "python_syntax": syntax, "functional_correctness": "not_tested"}


def save_json(path, data):
    # 同时保留模型原文和校验结果，便于比较 Prompt 修改前后的真实表现。
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 保持中文可读；backslashreplace 将无法编码的代理码点
    # 写成 JSON 可表示的转义，读回后仍能保留原值，而不是静默丢弃异常字符。
    # 同名文件会覆盖，比较多次实验时应指定不同的 --output。
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
                    errors="backslashreplace")


# 2. 应用流程：SDK 请求 → 累计正文 → 校验 → 保存 → 下一轮。
def conversation(args, base_url, environment):
    # args 保存命令行选项；base_url 是本次服务的实际地址；environment 写入记录。
    # 服务端不保存对话。每次请求都发送完整 messages，模型才能看到此前问答。
    # system 放在最前面，让后续多轮继续遵循相同角色和格式约定。
    system = {"role": "system", "content": SYSTEM_PROMPT + (FORMAT_PROMPT if args.structured else "")}
    messages, turn = [system], 0
    prompt = EXAMPLE_PROMPT if args.example else args.prompt
    # base_url 指向本脚本的本地服务，不会调用云端 OpenAI；api_key 仅为 SDK 占位值。
    # 关闭自动重试，使一次实验的失败和耗时可以直接观察。
    with OpenAI(base_url=base_url, api_key="local-demo", timeout=180, max_retries=0) as client:
        while True:
            turn += 1
            messages.append({"role": "user", "content": prompt})
            # CLI 中的 max_new_tokens 对应 Chat Completions 请求中的 max_tokens。
            # stream 决定一次返回完整响应，还是逐段返回正文；两条路径最后统一校验。
            request = dict(model="Qwen2.5-1.5B-Instruct", messages=messages,
                           max_tokens=args.max_new_tokens, temperature=0, stream=args.stream)
            if args.stream:
                # 请求结束时附带 token 用量；该事件没有正文，不能按普通 delta 读取。
                request["stream_options"] = {"include_usage": True}
            text, reason, usage, error, first_text = "", None, None, None, None
            # 计时包含请求、生成、接收和正文打印，不包含模型加载、后续校验及写文件。
            start = time.perf_counter()
            try:
                response = client.chat.completions.create(**request)
                if args.stream:
                    with response:
                        for chunk in response:
                            if chunk.usage is not None:
                                usage = chunk.usage.model_dump()
                            if not chunk.choices:  # usage 事件可能没有正文。
                                continue
                            choice = chunk.choices[0]
                            # 一个 delta 可能包含零个或多个文本字符，不一定对应一个 token。
                            # 必须先累加为完整正文，不能对每个不完整片段直接解析 JSON。
                            piece = choice.delta.content or ""
                            if piece and first_text is None:
                                # 首个非空文本片段的到达时间含通信和分词等开销，
                                # 也受 streamer 缓冲影响，不是纯 Prefill 或首 token 时间。
                                first_text = time.perf_counter() - start
                            text += piece
                            print(piece, end="", flush=True)
                            if choice.finish_reason is not None:
                                reason = choice.finish_reason
                else:
                    # API 外层响应已由 SDK 解析；message.content 仍是模型生成的字符串。
                    # 即使 API 响应合法，content 内的 JSON 也可能不合法。
                    text = response.choices[0].message.content or ""
                    reason = response.choices[0].finish_reason
                    usage = response.usage.model_dump() if response.usage else None
                    print(text, end="", flush=True)
            except (APIError, UnicodeError, KeyboardInterrupt) as exc:
                # 将 SDK、编码或生成期间中断等异常记入本轮；聊天模式还能继续输入。
                error = {"type": type(exc).__name__, "message": str(exc)}
            elapsed = time.perf_counter() - start
            # stop 只表示模型自然结束；length 表示达到上限。先排除不完整响应，
            # 再检查格式和代码语法，避免把截断内容当作可用结果。
            complete = not error and reason == "stop" and bool(text.strip())
            if error:
                result = {"status": "api_error", "error": error}
            elif not complete:
                detail = (f"已达到 {args.max_new_tokens} tokens 输出上限；可缩短要求或增加 --max-new-tokens"
                          if reason == "length" else "回答为空或未正常结束，不读取部分输出")
                result = {"status": "not_complete", "detail": detail}
            else:
                # --structured 开启时才解析模型正文；普通文本回答不要求 JSON。
                result = validate_text(text) if args.structured else {"status": "text_received"}
            output = (args.output.with_name(f"{args.output.stem}.turn{turn:02d}.json")
                      if args.chat else args.output)
            # model_json_schema() 导出字段约定，不是模型回答；parsed 则是校验后的数据。
            # 每轮独立保存，raw_text 始终保留原样，便于复核失败原因。
            save_json(output, {
                "record_origin": "live_model_call", "constraint_mode": "prompt_only",
                "environment": environment, "base_url": base_url, "request": request,
                "raw_text": text, "finish_reason": reason, "usage": usage, "result": result,
                "total_seconds": elapsed, "first_text_seconds": first_text,
                "json_schema": CodeResult.model_json_schema() if args.structured else None,
            })
            print(f"\n状态：{result['status']}；耗时：{elapsed:.3f}s；记录：{output}")
            if "error" in result:
                print(result["error"])
            if "detail" in result:
                print(result["detail"])
            if result["status"] == "schema_valid":
                syntax = result["python_syntax"]
                print("JSON/Schema：通过；Python 语法：", syntax["status"])
                if syntax["status"] == "syntax_error":
                    print("代码语法未通过：", syntax["error"])
                else:
                    print("语法通过；代码未执行，功能正确性尚未验证。")
                print("校验后读取的函数名：", result["parsed"]["function_name"])
                print("代码预览（只展示，不执行）：")
                print(result["parsed"]["code"])  # JSON 解析已将转义的换行还原为真实换行。
            if not args.chat:
                # 单轮的退出码反映该轮是否通过：完整文本或结构化且语法通过为 0。
                # 聊天模式的正常退出不代表每轮成功，应查看各轮记录中的状态。
                success = (result["status"] in ("text_received", "schema_valid")
                           and result.get("python_syntax", {}).get("status") != "syntax_error")
                return 0 if success else 1
            if complete:  # 格式失败也保留，下一轮可以要求修正；截断和网络错误不保留。
                messages.append({"role": "assistant", "content": text})
            else:
                # 本轮只追加过一条 user 消息；移除它即可恢复调用前的历史。
                messages.pop()
                print("本轮未完成，未加入历史。")
            while True:
                # 空行、/reset 和无效编码不会发请求，也不增加轮次；有效要求才进入下一轮。
                try:
                    prompt = input("\n继续提要求（/reset 清空，/exit 退出）：").strip()
                    prompt.encode("utf-8")  # 拒绝终端解码遗留的非法字符，不静默修改用户输入。
                except UnicodeError:
                    print("输入含有无法编码的字符，本条未发送。请重新输入；退出请重新键入 /exit。")
                    continue
                except (EOFError, KeyboardInterrupt):
                    return 0
                if prompt == "/exit":
                    return 0
                if prompt == "/reset":
                    # 清空的是本次会话历史，保留 system 规则；不重新加载模型。
                    messages = [system]
                    print("历史已清空，请输入新任务。")
                elif prompt:
                    break


# 3. HTTP / Transformers：请求验证、推理和响应封装。
# 下面的 Message / ChatRequest 校验 HTTP 请求，与上面的模型输出 CodeResult 是两回事。
class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    # 对应客户端的 stream_options；只实现流式结束时返回用量这一项。
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


class ChatRequest(BaseModel):
    # FastAPI 在进入处理函数前验证请求。模型名、角色、长度等不符合约定，
    # 或传入未实现字段时返回 422；输入 token 总数还需在分词后另行检查。
    model_config = ConfigDict(extra="forbid")  # 未实现的 response_format 等字段返回 422。
    model: Literal["Qwen2.5-1.5B-Instruct"]
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=512)
    temperature: Literal[0] = 0
    stream: bool = False
    stream_options: StreamOptions | None = None


def create_api(model, tokenizer, torch):
    # FastAPI 接收 SDK 发来的请求，再调用第一周使用的 Transformers generate。
    # 服务只处理一个生成请求；锁被占用时返回 429，避免同时争用同一个模型。
    # model/tokenizer 在 main 中加载一次，通过闭包供后续请求复用。
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    api, busy = FastAPI(), threading.Lock()

    @api.post("/v1/chat/completions")
    def completions(req: ChatRequest):
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "请等待上一条请求完成")
        try:
            # 聊天模板把角色与正文转换为模型所需的标记；分词时不再重复添加特殊 token。
            prompt = tokenizer.apply_chat_template([m.model_dump() for m in req.messages],
                                                   tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            # 计入完整历史及聊天模板标记，不是最后一条问题的字符数；张量与模型在同一设备。
            input_count = inputs.input_ids.shape[1]
            if input_count > 2048:
                raise HTTPException(400, "输入超过 2048 tokens，请 /reset 或缩短问题")
            # do_sample=False 使用 greedy decoding；KV cache 用于本次生成过程。
            # 这里只控制生成长度等参数，没有实现 JSON Schema 约束解码。
            # 每次请求重新编码完整历史；use_cache=True 不表示跨轮复用缓存或自动记忆会话。
            config = GenerationConfig(max_new_tokens=req.max_tokens, do_sample=False, use_cache=True,
                                      eos_token_id=model.generation_config.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True,
                                           timeout=0.2) if req.stream else None
            # skip_prompt 避免重复输出输入文本；0.2 秒是队列轮询等待时间，
            # 不是整个请求的超时，队列暂时为空时 HTTP 侧会继续等待。
        except BaseException:
            # 分词或请求准备失败也要释放锁，否则后续请求会一直收到 429。
            busy.release()
            raise
        result, cancelled = {}, threading.Event()
        # result 仅属于当前请求，在线程之间传递最终 token IDs 或异常信息；
        # cancelled 传递取消信号。读取最终结果前会等待生成线程结束。

        class Cancelled(StoppingCriteria):
            # 客户端断开或停止读取后，让生成线程在后续 token 步骤检查退出信号。
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()

        def generate():
            try:
                with torch.inference_mode():
                    # generate 返回“输入 + 新生成”的 token IDs，因此切掉输入部分。
                    result["ids"] = model.generate(**inputs, generation_config=config, streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([Cancelled()]))[0, input_count:].tolist()
            except Exception as exc:
                result["error"] = str(exc)
                if streamer is not None:
                    # 推理失败也要通知读取端结束，否则客户端可能一直等待下一段文本。
                    streamer.end()
            finally:
                # 无论推理成功或失败，都解除模型占用，允许下一次请求进入。
                busy.release()

        # 同一次请求的所有流式事件共用 id、created 和 model，便于对应到同一个回答。
        common = dict(id="chatcmpl-" + uuid.uuid4().hex, created=int(time.time()), model=req.model)

        def metadata():
            # token 数来自真实 IDs（含 EOS），不能用字符数或流式事件数替代。
            # EOS 对应 stop，但模型主动结束并不保证生成的代码完整或正确。
            ids, eos = result["ids"], config.eos_token_id
            # 模型配置可能用一个或多个 EOS ID；保留 EOS 参与 token 用量统计，
            # 但文本解码时会跳过特殊 token，因此文本长度与 token 数不相等。
            reason = "stop" if ids and ids[-1] in (eos if isinstance(eos, list) else [eos]) else "length"
            return reason, dict(prompt_tokens=input_count, completion_tokens=len(ids),
                                total_tokens=input_count + len(ids))

        if not req.stream:
            # 非流式：等待整段生成完成，将正文放入 choices[0].message.content。
            generate()
            if "error" in result:
                raise HTTPException(500, result["error"])
            reason, usage = metadata()
            return {**common, "object": "chat.completion", "usage": usage, "choices": [{
                "index": 0, "finish_reason": reason, "message": {"role": "assistant",
                "content": tokenizer.decode(result["ids"], skip_special_tokens=True)}}]}

        # 流式：工作线程生成 token，TextIteratorStreamer 把解码后的文本放进队列；
        # HTTP 响应协程同时从队列取出文本发送，客户端因此能边生成边看到结果。
        worker = threading.Thread(target=generate, daemon=True)
        try:
            worker.start()
        except BaseException:
            # 若线程本身未能启动，generate 的 finally 不会运行，需要在这里释放锁。
            busy.release()
            raise
        sentinel = object()

        def next_piece():
            # None 表示暂时没有文本，sentinel 才表示流已结束，二者不能混淆。
            try:
                return next(streamer, sentinel)
            except queue.Empty:
                return None

        def event(choices, **extra):
            # SSE 每条事件以 data: 开头、空行结束；SDK 从 JSON 事件中读取 delta。
            return "data: " + json.dumps({**common, "object": "chat.completion.chunk",
                                         "choices": choices, **extra}, ensure_ascii=False) + "\n\n"

        async def events():
            try:
                while True:  # generate 尚在进行时发送文本，不是生成完后再切片。
                    piece = await asyncio.to_thread(next_piece)
                    if piece is sentinel:
                        break
                    if piece:
                        yield event([dict(index=0, delta={"content": piece}, finish_reason=None)])
                # 等生成线程写好最终 IDs，再计算结束原因和 token 用量。
                # to_thread 避免等待队列或线程时阻塞 HTTP 服务的异步事件循环。
                await asyncio.to_thread(worker.join)
                if "error" in result:
                    # 已开始发送流式响应时，通过流中的 error 事件报告生成异常。
                    yield "data: " + json.dumps({"error": {"message": result["error"]}}) + "\n\n"
                    return
                reason, usage = metadata()
                yield event([dict(index=0, delta={}, finish_reason=reason)])
                if req.stream_options and req.stream_options.include_usage:
                    # 用量单独发一条 choices=[] 的事件，客户端需跳过它的正文读取。
                    yield event([], usage=usage)
                # [DONE] 是传输结束标记，不属于模型正文，也不参与 token 统计。
                yield "data: [DONE]\n\n"
            finally:
                cancelled.set()

        return StreamingResponse(events(), media_type="text/event-stream")

    return api


def main():
    # 解析命令行选项，加载模型并启动服务后调用 conversation。
    # 终端无法显示的字符用转义表示，避免打印异常字符时整个程序退出。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct", help="本地 1.5B 模型目录")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", default=USER_PROMPT)
    prompts.add_argument("--example", action="store_true", help="使用内置参考 Prompt，并启用结构化校验")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/run.json"))
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 512 or not args.prompt.strip():
        parser.error("输出长度须在 1–512 之间，Prompt 不能为空")
    args.structured = args.structured or args.example
    try:
        args.prompt.encode("utf-8")
    except UnicodeError:
        parser.error("Prompt 含有无法编码的字符，请重新输入")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请检查 GPU 版 PyTorch，或使用 --device cpu")
    torch.set_num_threads(8)
    # 设置 CPU 运算线程数；不会限制 CUDA 核心数，也不是模型性能基准配置。
    # 沿用第一周的精度选择：CPU 用 FP32，CUDA 优先 BF16，否则 FP16。
    dtype = torch.float32 if args.device == "cpu" else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    print(f"加载模型：{args.model}；device={args.device}；dtype={dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # local_files_only=True 只读取已有模型，不会在启动时下载缺失的权重。
    # eval 切换模型行为；inference_mode 作用于实际推理，减少梯度跟踪开销。
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
        attn_implementation="eager", local_files_only=True).to(args.device).eval()
    environment = {"model": args.model, "device": args.device, "dtype": str(dtype),
        "python": platform.python_version(), "torch": torch.__version__,
        **{name: version(name) for name in ["transformers", "openai", "pydantic", "fastapi", "uvicorn"]}}

    # 主线程运行客户端，后台线程运行本地 HTTP 服务，两者通过真实 HTTP 请求通信。
    # 127.0.0.1 只接受本机连接；端口 0 让系统分配空闲端口，避免固定端口冲突。
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        base_url = f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
        server = uvicorn.Server(uvicorn.Config(create_api(model, tokenizer, torch), log_level="error"))
        worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        # 预先绑定的 socket 直接交给 Uvicorn 使用，避免选好端口后又被别的进程占用。
        worker.start()
        # 模型已经加载完毕；此处的 15 秒只等待 HTTP 监听就绪，防止请求发得过早。
        deadline = time.monotonic() + 15
        try:
            while not server.started:
                if not worker.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("本地 HTTP 服务未能启动，请检查终端日志")
                time.sleep(0.05)
            print("本地 API 已就绪：", base_url, flush=True)
            return conversation(args, base_url, environment)
        finally:
            # 正常退出或发生异常都通知本次 HTTP 服务关闭；进程退出后释放模型资源。
            server.should_exit = True
            # 先请求服务退出，再等待线程收尾；此处不删除模型文件或运行记录。
            worker.join(timeout=5)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"运行失败：{exc}") from None
