"""第三周：Agent v0 —— 模型 → 工具 → Observation → 最终回答 的最小闭环。

先用脚本化模型验证循环与工具（无需 GPU、无需下载模型）：
    python agent_v0.py --selftest

再接入真实模型（二选一）：
    python agent_v0.py --backend hf --model ./models/Qwen2.5-1.5B-Instruct --device cpu \
        --task "读取 test_report.txt，计算测试通过率" --output results/run.json
    python agent_v0.py --backend openai --base-url http://localhost:8000/v1 --model <模型名> --task "..."

hf 后端默认贪心解码（同样输入、同样输出，方便复现和调试）；加 --sample 开启采样
（可配 --temperature / --top-p / --top-k），多跑几次观察成功率的波动。

llm(messages) -> str 是唯一的模型接口：换模型、换服务（本地 transformers、vLLM）只改这一层。
"""
import argparse
import ast
import json
import operator
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

WORKDIR = Path("workspace")   # read_file 只能读取这个目录内的文件
MAX_OBS_CHARS = 2000          # Observation 长度上限，防止工具输出撑爆上下文


# ---------------------------------------------------------------- 1. 工具
# 每个工具 = 一个 Python 函数 + 一份参数模型（Pydantic）。
# 同一份参数模型两处使用：导出 JSON Schema 告诉模型（见 build_system_prompt）；
# 校验模型给出的 arguments（见 execute）。strict=True 关闭隐式类型转换（比如数字 12
# 不会被自动转成字符串 "12"）；extra="forbid" 禁止模型多传约定之外的字段。
class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    # max_length 防止构造超长表达式（比如几百个数字连乘出一个巨大整数）占满 CPU/内存；
    # calculator 不放行 **，但加减乘除本身在表达式足够长时也能造成明显的计算开销。
    expression: str = Field(min_length=1, max_length=200, description="算术表达式，只含数字、+ - * / 与括号")


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, description="工作目录内的相对路径，例如 test_report.txt")


class WordCountArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, description="工作目录内的相对路径，例如 test_report.txt")


# ast 节点类型 -> 对应运算符函数；只登记这四种，calculator 里没登记的语法一律拒绝。
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculator(expression: str):
    # 不用 eval（会执行任意 Python 代码，存在命令注入风险）：先把表达式解析成语法树，
    # 再递归求值，只放行“数字常量、加减乘除、取负号”这几类节点，其余一律抛异常。
    def ev(n):
        t = type(n)
        if t is ast.Constant and type(n.value) in (int, float):
            return n.value
        if t is ast.BinOp and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if t is ast.UnaryOp and type(n.op) is ast.USub:
            return -ev(n.operand)
        raise ValueError("表达式含有不支持的语法")
    return ev(ast.parse(expression, mode="eval").body)


def read_file(path: str) -> str:
    # 把用户给的相对路径解析成绝对路径，再检查它是否仍落在 WORKDIR 内部，
    # 从而拒绝 "../secret.txt" 这类试图跳出工作目录的路径（路径穿越攻击）。
    root = WORKDIR.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise PermissionError(f"只允许读取工作目录内的文件，请使用工作目录内的相对路径（不要用 / 开头的绝对路径），"
                               f"例如 test_report.txt；收到的路径是 {path!r}")
    if not target.is_file():                     # 报错里只写用户给的相对路径，不暴露绝对路径
        raise FileNotFoundError(f"文件不存在：{path}")
    return target.read_text(encoding="utf-8")


def word_count(path: str) -> dict:
    # 自选只读工具示例：统计工作目录内文件的行数/词数/字符数，复用 read_file 做安全校验。
    # 只是示例，换成别的只读工具（比如列目录、解析某种格式）完全没问题。
    text = read_file(path)
    return {"lines": len(text.splitlines()), "words": len(text.split()), "chars": len(text)}


@dataclass
class Tool:
    fn: Callable            # 实际执行的函数
    args: type[BaseModel]   # 对应的参数模型，用于校验 + 导出 JSON Schema
    description: str        # 给模型看的工具说明


# 工具注册表：新增工具只需要在这里加一行 Tool(函数, 参数模型, 描述)，
# build_system_prompt（生成工具列表）和 execute（校验+调用）都从这张表里查。
TOOLS: dict[str, Tool] = {
    "calculator": Tool(calculator, CalculatorArgs, "计算算术表达式，返回数值结果。"),
    "read_file": Tool(read_file, ReadFileArgs, "读取工作目录内的文本文件，返回文件内容。"),
    "word_count": Tool(word_count, WordCountArgs, "统计工作目录内文本文件的行数、词数与字符数。"),
}


# ---------------------------------------------------------------- 2. Prompt 与决策协议
# 整个 Agent 的“协议”就靠这段系统提示词口头约定：模型只能从两种动作里选一种，
# 而且每轮只能输出一个 JSON 对象——不依赖任何服务端的强制解码（constrained decoding），
# 所以模型不一定会遵守，这也是后面 parse_decision/execute 里要做容错和校验的原因。
SYSTEM_TEMPLATE = """你是一个可以使用工具的助手。
可用工具（name、description 与参数的 JSON Schema）：
{tools}

每一轮只输出一个 JSON 对象，不要输出其他文字：
- 需要调用工具：{"action":"tool","name":"<工具名>","arguments":{...}}
- 可以给出回答：{"action":"final","answer":"<最终回答>"}
工具的执行结果会以 "Observation: ..." 的形式返回。拿到足够信息后再给出 final。"""


def build_system_prompt() -> str:
    # 把 TOOLS 注册表转换成 [{"name", "description", "parameters"}, ...]，
    # parameters 就是 Pydantic 参数模型导出的 JSON Schema，模型靠这个知道每个工具接受什么参数。
    specs = [{"name": n, "description": t.description, "parameters": t.args.model_json_schema()}
             for n, t in TOOLS.items()]
    return SYSTEM_TEMPLATE.replace("{tools}", json.dumps(specs, ensure_ascii=False))


# ToolCall / Final 是模型每一轮输出必须落在的两种合法结构之一。
class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["tool"]
    name: str
    arguments: dict[str, Any] = {}


class Final(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["final"]
    answer: str


# 判别联合（discriminated union）：按 JSON 里的 "action" 字段值，自动决定按 ToolCall
# 还是 Final 的规则校验，比手写 if/else 分别校验两种结构更简洁、报错信息也更准确。
Decision = Annotated[Union[ToolCall, Final], Field(discriminator="action")]
_decision = TypeAdapter(Decision)


def parse_decision(text: str) -> Union[ToolCall, Final]:
    """把模型原文解析成 ToolCall 或 Final；失败抛 ValueError（含 pydantic 的 ValidationError）。"""
    # 模型经常会在 JSON 前后多写几句话或代码围栏，这里只截取第一个 { 到最后一个 } 之间
    # 的内容去解析，能兼容这类“正文+JSON”的输出；真正的格式错误交给下面的 pydantic 校验判断。
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("输出中没有 JSON 对象")
    return _decision.validate_json(text[start:end + 1])


# ---------------------------------------------------------------- 3. 工具执行：校验 → 执行 → 统一成 Observation
# 这一层的核心原则：execute() 面对任何输入（未知工具名、参数错误、工具内部异常）
# 都只返回一个 dict，绝不抛异常——工具失败是 Agent Loop 里的正常情况，不能让程序崩溃。
def _err(kind: str, message: str) -> dict:
    return {"status": "error", "type": kind, "message": message}


def _fmt(e: dict) -> str:                     # 把 Pydantic 的一条错误写成“字段: 原因”
    return f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}"


def execute(name: str, arguments: dict) -> dict:
    """永远返回 dict，不向循环抛异常。"""
    # 第一步：工具名是否存在。
    tool = TOOLS.get(name)
    if tool is None:
        names = list(TOOLS)
        return _err("unknown_tool", f"未知工具 {name!r}，可用工具：{names}")
    # 第二步：参数校验——用模型自己给的 arguments 去套对应的 Pydantic 参数模型，
    # 缺参数/多参数/类型错误都会在这里被 ValidationError 拦下，不会传到工具函数里。
    try:
        args = tool.args.model_validate(arguments)
    except ValidationError as exc:
        msg = "; ".join(_fmt(e) for e in exc.errors())
        return _err("invalid_arguments", msg)
    # 第三步：真正调用工具函数，工具内部抛的任何异常（文件不存在、除零、权限错误……）
    # 都在这里统一接住，转成 execution_error，不会向上传播打断 Agent Loop。
    try:
        result = tool.fn(**args.model_dump())
        return {"status": "ok", "result": result}
    except Exception as exc:
        return _err("execution_error", f"{type(exc).__name__}: {exc}")


def observation_text(obs: dict) -> str:
    # 把 execute() 的返回值序列化成 "Observation: {...}" 文本塞回对话历史。
    # 超过 MAX_OBS_CHARS 就截断：工具返回内容不可控（比如读到一个很大的文件），
    # 不截断会一路撑爆后续所有轮次的上下文长度。
    text = "Observation: " + json.dumps(obs, ensure_ascii=False, default=str)
    return text if len(text) <= MAX_OBS_CHARS else text[:MAX_OBS_CHARS] + "…(已截断)"


# ---------------------------------------------------------------- 4. Agent Loop
# 全部代码里最核心的函数：每一轮都是“调模型 → 解析决策 → 执行/终止 → 判断是否继续”，
# 循环本身不关心模型是什么、工具是什么，只负责推进状态和判断何时停下来。
def run_agent(task: str, llm: Callable[[list], str], max_steps: int = 6, max_failures: int = 2) -> dict:
    """返回 {"status", "answer", "steps", "messages", "trace"}。
    status: ok | max_steps | no_progress | too_many_failures
    """
    # 初始状态：标准的 chat 格式，system 放协议约定+工具列表，user 放本次任务。
    messages = [{"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": task}]
    trace, failures, seen = [], 0, Counter()
    # trace：每一步的完整记录（原文/决策/Observation/耗时），存到 --output 的 JSON 里，
    #        是排查“Loop 卡在哪一步、为什么”的主要依据。
    # failures：累计失败次数（格式错误 或 工具执行报错），达到预算就停止，防止无限重试。
    # seen：记录每个 (工具名, 参数) 组合出现的次数，用于判断“同一个调用是否在原地打转”。

    def finish(status, answer=None):
        return {"status": status, "answer": answer, "steps": len(trace), "messages": messages, "trace": trace}

    for step in range(1, max_steps + 1):
        # 第一步：把当前完整对话历史交给模型，拿到这一轮的原始输出。
        t0 = time.perf_counter()
        raw = llm(messages)
        record = {"step": step, "raw": raw, "latency_s": round(time.perf_counter() - t0, 3)}
        trace.append(record)
        messages.append({"role": "assistant", "content": raw})

        # 第二步：把原文解析成结构化决策。模型不一定会守规矩，解析失败是常态而不是异常：
        # 累计一次失败，把纠错提示作为新的 "user" 消息追加进去，让模型下一轮有机会自己改正；
        # 只有连续失败次数超过预算才真正终止，避免一次偶然的格式错误就放弃整个任务。
        try:
            decision = parse_decision(raw)
        except ValueError as exc:
            failures += 1
            record["error"] = f"format_error: {exc}"
            if failures > max_failures:
                return finish("too_many_failures")
            messages.append({"role": "user",
                             "content": "Observation: 输出格式错误，请只输出一个符合要求的 JSON 对象。"})
            continue

        # 第三步：按解析出的动作分流。final 直接结束整个 Loop；tool 则调用 execute()
        # 拿到真实的执行结果（不是模型自己编的），并把 Observation 追加回对话历史，
        # 这样下一轮模型才能“看到”工具真正返回了什么，据此决定下一步怎么做。
        if decision.action == "final":
            record["decision"] = {"action": "final"}
            return finish("ok", decision.answer)
        record["decision"] = {"action": "tool", "name": decision.name, "arguments": decision.arguments}
        obs = execute(decision.name, decision.arguments)
        record["observation"] = obs
        failures = failures + 1 if obs["status"] == "error" else 0   # 工具调用成功则清零失败计数
        messages.append({"role": "user", "content": observation_text(obs)})

        # 第四步：两条停止条件。一是失败预算耗尽（跟第二步共用同一个 failures 计数器）；
        # 二是同一个 (工具名, 参数) 组合被连续调用到第 3 次，判定为“卡住了、没有新进展”，
        # 避免模型陷入重复调用同一个操作却不推进任务的死循环。
        if failures > max_failures:
            return finish("too_many_failures")
        seen[(decision.name, json.dumps(decision.arguments, sort_keys=True, ensure_ascii=False))] += 1
        if max(seen.values()) >= 3:
            return finish("no_progress")
    return finish("max_steps")   # 达到步数上限仍未 final，也要有干净的收尾


# ---------------------------------------------------------------- 5. 模型后端：llm(messages) -> str
# run_agent() 只依赖 llm(messages) -> str 这一个接口，跟模型/服务的具体实现完全解耦：
# 下面三种实现（脚本化、本地 transformers、OpenAI 兼容 API）随便换，Agent Loop 不用改一行。
class ScriptedLLM:
    """脚本化模型：按顺序返回预设输出，用于确定性地测试循环，不依赖真实模型。"""
    def __init__(self, outputs):
        self.outputs, self.i = list(outputs), 0

    def __call__(self, messages):
        assert self.i < len(self.outputs), "脚本化模型的输出已用完"
        self.i += 1
        return self.outputs[self.i - 1]


class _BalancedJSONStop:
    """生成到第一个闭合的顶层 JSON 对象就停，防止小模型在合法输出后继续自问自答、
    臆造后续的 Observation 和决策（真实工具调用交给 Agent Loop 的 execute()，不该由模型自己编）。

    做法：每次都把“新生成的这部分”解码出来，从第一个 { 开始数括号深度，深度回到 0
    （也就是遇到与第一个 { 配对的那个 }）就立刻停止生成，不管 max_new_tokens 还没用完。
    """
    def __init__(self, tokenizer, prompt_len):
        self.tokenizer, self.prompt_len = tokenizer, prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        start = text.find("{")
        if start < 0:
            return False
        depth = 0
        for ch in text[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False


def make_hf_llm(model_dir, device, max_new_tokens, sample=False, temperature=0.7, top_p=0.8, top_k=20):
    """本地 transformers 推理：加载一次模型，返回的 llm() 每次都跑一次完整的 generate。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 GPU 版 PyTorch，或使用 --device cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    # CPU 用 bf16：内存只要 float32 的一半，7B/14B 这类模型更安全；GPU 上先确认硬件真的
    # 支持 bf16（Ampere 及更新架构），不支持则退回 fp16，避免在老显卡上报错或跑得很慢。
    dtype = torch.bfloat16 if device == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).to(device).eval()
    # sample=False（默认）：贪心解码，每步都选概率最高的 token，同样的输入永远得到同样的
    # 输出，方便复现和调试；sample=True：按 temperature/top_p/top_k 做随机采样，同样的
    # 输入每次跑出来可能不一样，适合多跑几次、观察成功率的波动。
    # 贪心解码时不传 temperature/top_p/top_k：这几个参数只在采样时起作用，硬传会触发
    # transformers 的告警（"only used in sample-based generation modes"）。
    gen_kwargs = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k} if sample \
        else {"do_sample": False}

    def llm(messages):
        # apply_chat_template 把 [{"role":..,"content":..}, ...] 转成模型认识的带特殊
        # 标记的纯文本；add_generation_prompt=True 会在末尾加上“该模型回复了”的引导标记。
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        stopping = StoppingCriteriaList([_BalancedJSONStop(tok, inputs.input_ids.shape[1])])
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, stopping_criteria=stopping, **gen_kwargs)
        # generate 返回的是“输入+新生成”拼在一起的完整 token 序列，所以要切掉前面输入
        # 那部分长度（inputs.input_ids.shape[1]），只解码新生成的这一段。
        return tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return llm


def make_openai_llm(base_url, model, max_new_tokens, api_key="local-demo"):
    """OpenAI 兼容 API（比如 vLLM 起的服务）：走标准 Chat Completions 接口。
    temperature=0 相当于贪心解码；这个后端目前没有暴露 --sample 开关。"""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180, max_retries=0)

    def llm(messages):
        r = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=max_new_tokens)
        return r.choices[0].message.content or ""
    return llm


# ---------------------------------------------------------------- 6. 自测：固定任务集（脚本化模型）
# 用 ScriptedLLM 按预设顺序“扮演”模型的输出，不需要真实模型或 GPU，就能确定性地把
# Agent Loop 的每条分支（成功、格式错误后修正、参数错误后修正、无进展、步数上限……）都跑一遍。
def call(name, **arguments):
    return json.dumps({"action": "tool", "name": name, "arguments": arguments}, ensure_ascii=False)


def final(answer):
    return json.dumps({"action": "final", "answer": answer}, ensure_ascii=False)


def selftest():
    global WORKDIR
    with tempfile.TemporaryDirectory() as tmp:   # 用临时目录做 read_file 的工作目录，测试结束自动清理
        WORKDIR = Path(tmp)
        (WORKDIR / "test_report.txt").write_text("passed=47\nfailed=3\n", encoding="utf-8")
        cases = [
            # (名称, 脚本化输出, 期望 status, 期望 answer 或 None, 期望步数, 参数)
            ("T1 单步计算", [call("calculator", expression="1024*1024"), final("1048576")], "ok", "1048576", 2, {}),
            ("T2 两步调用", [call("read_file", path="test_report.txt"), call("calculator", expression="47/(47+3)*100"),
                        final("94%")], "ok", "94%", 3, {}),
            ("T3 无需工具", [final("Agent Loop 是反复调用模型并执行工具直到给出回答的循环。")], "ok", None, 1, {}),
            ("T4a 参数非法后修正", [call("calculator", expr="1+1"), call("calculator", expression="1+1"), final("2")],
             "ok", "2", 3, {}),
            ("T4b 未知工具后修正", [call("run_tests", target="tests"), call("calculator", expression="2*3"), final("6")],
             "ok", "6", 3, {}),
            ("T4c 除零后修正", [call("calculator", expression="1/0"), call("calculator", expression="1/1"), final("1")],
             "ok", "1", 3, {}),
            ("T5 重复调用（无进展）", [call("calculator", expression="1+1")] * 6, "no_progress", None, 3, {}),
            ("T6 步数上限", [call("calculator", expression=f"1+{i}") for i in range(6)], "max_steps", None, 3,
             {"max_steps": 3}),
            ("T7 格式持续错误", ["我来想想", "好的", "嗯", "呃"], "too_many_failures", None, 3, {}),
            ("T8 越权读文件后失败", [call("read_file", path="../.env")] * 3, "too_many_failures", None, 2,
             {"max_failures": 1, "max_steps": 5}),
        ]
        ok_all = True
        for name, outputs, status, answer, steps, kw in cases:
            try:
                result = run_agent("测试任务", ScriptedLLM(outputs), **kw)
            except Exception as exc:              # 意外异常也显示为 FAIL，而不是让整个自测崩溃退出
                ok_all = False
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
                continue
            ok = result["status"] == status and (answer is None or result["answer"] == answer) and result["steps"] == steps
            ok_all &= ok
            print(f"{'PASS' if ok else 'FAIL'}  {name}: status={result['status']} steps={result['steps']}"
                  + ("" if ok else f"  (期望 status={status} steps={steps} answer={answer})"))
        # 安全：calculator 不接受任何非算术语法
        for bad in ["__import__('os').system('echo hi')", "2**9999999", "abs(-1)"]:
            try:
                r = execute("calculator", {"expression": bad})
            except Exception as exc:
                r = {"status": "crashed", "type": type(exc).__name__}
            ok = r["status"] == "error"
            ok_all &= ok
            print(f"{'PASS' if ok else 'FAIL'}  拒绝非法表达式 {bad[:24]!r}: {r.get('type')}")
        # 自选工具（word_count）冒烟测试
        wc = execute("word_count", {"path": "test_report.txt"})
        ok = wc["status"] == "ok" and wc["result"]["lines"] == 2
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  自选工具 word_count: {wc}")
    print("全部通过" if ok_all else "存在失败项")
    return ok_all


def main():
    # 命令行入口：--selftest 只跑自测（不需要模型）；否则按 --backend 选择 hf（本地
    # transformers）或 openai（兼容 API）构造 llm()，再交给 run_agent() 跑真实任务。
    ap = argparse.ArgumentParser(description="第三周 Agent v0")
    ap.add_argument("--selftest", action="store_true", help="用脚本化模型运行固定任务集")
    ap.add_argument("--backend", choices=["hf", "openai"], default="hf")
    ap.add_argument("--model", default="./models/Qwen2.5-1.5B-Instruct", help="hf: 本地目录；openai: 服务端模型名")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--task", default="计算 1024*1024")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--max-failures", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--workdir", default="workspace")
    ap.add_argument("--output", default="results/run.json")
    ap.add_argument("--sample", action="store_true", help="hf 后端开启采样（默认贪心解码，同样输入结果确定性不变）")
    ap.add_argument("--temperature", type=float, default=0.7, help="仅 --sample 时生效")
    ap.add_argument("--top-p", type=float, default=0.8, help="仅 --sample 时生效")
    ap.add_argument("--top-k", type=int, default=20, help="仅 --sample 时生效")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    global WORKDIR
    WORKDIR = Path(args.workdir)
    if not WORKDIR.exists():                          # 首次运行：自动准备演示用的工作目录
        WORKDIR.mkdir(parents=True)
        (WORKDIR / "test_report.txt").write_text("passed=47\nfailed=3\n", encoding="utf-8")
    llm = (make_hf_llm(args.model, args.device, args.max_new_tokens,
                        sample=args.sample, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
           if args.backend == "hf" else make_openai_llm(args.base_url, args.model, args.max_new_tokens))
    result = run_agent(args.task, llm, args.max_steps, args.max_failures)
    for rec in result["trace"]:                       # 打印调用链
        print(f"[step {rec['step']}] {rec.get('decision', rec.get('error'))}  -> {rec.get('observation', '')}")
    print("status:", result["status"], "| answer:", result["answer"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": args.task, **result}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
