#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第十周：量化前后的同口径对比。

一条命令跑完四步：健康检查 → 性能压测 → 精度评测 → Agent 指标，
结果统一写进 results/，供报告直接引用。只依赖 Python 标准库；
压测用 vllm bench serve，评测用 lm_eval，需要提前装好。

用法：
    python bench_quant.py --selftest                       # 不需要 GPU，先确认脚本本身没问题
    python bench_quant.py --model qwen3-8b-bf16 --weight-bits 16   # 测基线（服务已启动）
    python bench_quant.py --model qwen3-8b-fp8  --weight-bits 8    # 换成 FP8，其余参数不变
    python bench_quant.py --compare results/run_qwen3-8b-bf16_c1.json results/run_qwen3-8b-fp8_c1.json

两次运行除了模型之外，其余参数必须完全一致，--compare 会校验这一点。
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

RESULT_DIR = "results"

# 两次运行之间必须保持一致的字段；compare 时会校验
INVARIANT_KEYS = [
    "input_len", "output_len", "num_prompts", "concurrency",
    "temperature", "seed", "eval_tasks", "eval_limit", "num_fewshot",
]

# ====================================================================
# 默认参数（都可以用命令行覆盖）
# ====================================================================
DEFAULT_CONFIG = {
    "name": "",                               # 结果文件名前缀；留空则用「模型名_c并发数」
    "model": "",                              # 必填：服务的 --served-model-name
    "tokenizer": "/srv/spark/models/Qwen3-8B",  # 分词器目录；两次运行用同一份
    "base_url": "http://127.0.0.1:%s" % os.environ.get("PORT", "8100"),
    "weight_bits": 0,                         # 必填：16 / 8，用于屋顶线估算
    "param_count_b": 8.03,                    # 单位：十亿参数
    "mem_bandwidth_gb_s": 273.0,              # DGX Spark 的内存带宽
    # ---- 压测负载：两次运行必须一致 ----
    "input_len": 1024,
    "output_len": 128,
    "num_prompts": 32,
    "concurrency": 1,
    "temperature": 0.0,
    "seed": 1234,
    # ---- 精度评测：两次运行必须一致 ----
    "eval_tasks": "gsm8k,arc_challenge",
    "eval_limit": 100,
    "num_fewshot": 5,
    # ---- Agent 指标 ----
    "agent_trace": "",                        # 你们自己任务集跑出来的 trace 文件；留空则跳过
}


# ====================================================================
# 第 0 步：健康检查
# ====================================================================
def wait_for_server(base_url, model, timeout=300, warmup=3):
    """等服务起来，确认它认出了模型，并做几次预热请求。返回 True 表示可以开始压测。

    1) 轮询 {base_url}/v1/models，直到返回 200 或超时；
    2) 检查返回的 data 列表里有没有 model 这个 id，没有就报错退出
       （常见原因：--served-model-name 和 --model 写的名字不一致）；
    3) 发 warmup 次 chat 请求预热，不计时也不计入结果。
       第一次请求会触发编译与显存分配，不预热会让 TTFT 严重偏高。
    """
    deadline = time.time() + timeout
    models = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    models = json.load(r).get("data", [])
                    break
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(3)
    if models is None:
        sys.exit("服务在 %ds 内没有起来：%s" % (timeout, base_url))
    ids = [m.get("id") for m in models]
    if model not in ids:
        sys.exit("服务里没有模型 %r，实际是 %s。检查 --served-model-name 与 --model 是否一致"
                 % (model, ids))
    body = json.dumps({"model": model, "max_tokens": 8, "temperature": 0.0,
                       "messages": [{"role": "user", "content": "你好"}]}).encode()
    for _ in range(warmup):
        req = urllib.request.Request(
            base_url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            r.read()
    return True


# ====================================================================
# 第 1 步：性能压测
# ====================================================================
def build_bench_cmd(cfg):
    """拼出 vllm bench serve 的命令行。所有取值都从 cfg 里读，保证两次运行同口径。

    --ignore-eos 让每个请求都生成满 output_len，否则模型提前停止，两次运行的输出长度对不上；
    --tokenizer 在服务名不是 HF 仓库名时必须指定，否则客户端找不到分词器。
    """
    cmd = ["vllm", "bench", "serve",
           "--model", cfg["model"],
           "--base-url", cfg["base_url"],
           "--dataset-name", "random",
           "--random-input-len", str(cfg["input_len"]),
           "--random-output-len", str(cfg["output_len"]),
           "--num-prompts", str(cfg["num_prompts"]),
           "--max-concurrency", str(cfg["concurrency"]),
           "--seed", str(cfg["seed"]),
           "--temperature", str(cfg["temperature"]),
           "--ignore-eos",
           "--percentile-metrics", "ttft,tpot,itl",
           "--metric-percentiles", "50,99",
           "--save-result",
           "--result-filename",
           os.path.join(RESULT_DIR, "perf_%s.json" % cfg["name"])]
    if cfg.get("tokenizer"):
        cmd += ["--tokenizer", cfg["tokenizer"]]
    return cmd


def parse_bench_result(path):
    """读 vllm bench serve 存下来的 JSON，提取四个指标（延迟 ms，吞吐 tokens/s）。

    不同 vLLM 版本的字段名可能不同；找不到字段时抛出明确的错误，不静默返回 0。
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    want = {"ttft_p50": "median_ttft_ms", "ttft_p99": "p99_ttft_ms",
            "tpot_p50": "median_tpot_ms", "output_throughput": "output_throughput"}
    out = {}
    for k, field in want.items():
        if field not in d or d[field] is None:
            raise KeyError("结果文件里没有字段 %r，现有字段：%s" % (field, sorted(d)))
        out[k] = float(d[field])
    return out


def peak_memory_gb():
    """记录 nvidia-smi 看到的显存占用（GB）。失败时返回 None，不影响整个流程。

    两点局限，写报告时要说明：
    1) vLLM 起服务时就按 --gpu-memory-utilization 把内存池一次占满，
       所以这个数两次运行几乎相同，不能当作「量化省了多少内存」的依据。
       请改用服务日志里的 `Model loading took … GiB`（权重占用）
       和 `Available KV cache memory`（KV 可用空间）。
    2) GB10 上 nvidia-smi 的显存一栏是 N/A，这里会返回 None。
    """
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used",
               "--format=csv,noheader,nounits"]
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis:                       # 多卡共用机器：只看自己用的那几张
            cmd += ["-i", vis]
        out = subprocess.check_output(cmd, text=True, timeout=10)
        return max(float(x) for x in out.split()) / 1024.0
    except Exception:
        return None


# ====================================================================
# 第 2 步：精度评测
# ====================================================================
def build_eval_cmd(cfg):
    """拼出 lm_eval 的命令行。两次运行的取值必须一致，只有 model 可以不同。

    lm_eval 会把 --output_path 当作文件名前缀，实际写出的文件叫
    eval_<name>_<时间戳>.json，run_once 里用通配符去找。
    """
    margs = "model=%s,base_url=%s/v1/completions,num_concurrent=%d,max_retries=3" % (
        cfg["model"], cfg["base_url"], cfg["concurrency"])
    if cfg.get("tokenizer"):
        margs += ",tokenizer=%s" % cfg["tokenizer"]
    return ["lm_eval", "--model", "local-completions",
            "--model_args", margs,
            "--tasks", str(cfg["eval_tasks"]),
            "--num_fewshot", str(cfg["num_fewshot"]),
            "--limit", str(cfg["eval_limit"]),
            "--seed", str(cfg["seed"]),
            "--output_path", os.path.join(RESULT_DIR, "eval_%s.json" % cfg["name"])]


def parse_eval_result(path):
    """读 lm_eval 的输出，返回 {任务名: 主指标分数}。两次运行挑的是同一个主指标。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for task, metrics in data.get("results", {}).items():
        for key in ("exact_match,strict-match", "acc_norm,none", "acc,none"):
            if key in metrics:
                out[task] = round(float(metrics[key]), 4)
                break
    return out


# ====================================================================
# 第 3 步：Agent 指标
# ====================================================================
def agent_metrics(trace_path, schema=None):
    """统计工具调用成功率、结构化输出合法率、任务完成率。

    trace 文件格式由各组自定，每行一条 JSON，至少包含：
        {"task_id": ..., "steps": [...],
         "tool_calls": [{"args_valid": true/false, "error": null 或错误信息}],
         "final_output": ..., "success": true/false}

    三条口径要在报告里写清楚：
      1) 工具调用成功：参数通过校验 且 工具执行没抛异常，缺一不可；
      2) 结构化输出合法：final_output 能通过 JSON Schema 校验
         （没有 Schema 时只统计 json.loads 成功率，报告里要注明口径不同）；
      3) 任务完成：按各组自己任务集的判定标准。
    分母为 0 时返回 None 而不是 0.0，不把「没测」写成「0%」。
    """
    rows = []
    with open(trace_path, "r", encoding="utf-8") as f:   # 文件不存在时抛 FileNotFoundError
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def rate(num, den):
        return round(num / den, 4) if den else None

    calls = [c for r in rows for c in r.get("tool_calls", [])]
    n_ok = sum(1 for c in calls if c.get("args_valid") and not c.get("error"))

    def schema_ok(out):
        try:
            obj = json.loads(out) if isinstance(out, str) else out
        except (ValueError, TypeError):
            return False
        if schema is None:
            return True
        import jsonschema
        try:
            jsonschema.validate(obj, schema)
            return True
        except jsonschema.ValidationError:
            return False

    n_schema = sum(1 for r in rows if schema_ok(r.get("final_output")))
    n_succ = sum(1 for r in rows if r.get("success"))
    steps = [len(r.get("steps", [])) for r in rows]
    return {"tool_call_ok_rate": rate(n_ok, len(calls)),
            "schema_ok_rate": rate(n_schema, len(rows)),
            "task_success_rate": rate(n_succ, len(rows)),
            "avg_steps": round(sum(steps) / len(steps), 2) if steps else None,
            "n_tasks": len(rows), "n_tool_calls": len(calls)}


# ====================================================================
# 第 4 步：对比
# ====================================================================
def roofline_tok_s(param_count_b, weight_bits, bandwidth_gb_s):
    """解码速度的理论上限：带宽 ÷ 权重体积。"""
    weight_gb = param_count_b * 1e9 * weight_bits / 8 / 1e9
    return bandwidth_gb_s / weight_gb


def compare(base, quant):
    """对比两次运行，返回可直接写进报告的 dict。

    1) 先校验 INVARIANT_KEYS 里的字段两边是否一致，不一致就抛错并指出是哪一项；
    2) 性能按「量化 ÷ 基线」给倍数（延迟越小越好，吞吐越大越好）；
    3) 精度按「量化 − 基线」给百分点差值；
    4) 用 roofline_tok_s 算出两边各自的屋顶线与达成率。
    """
    for k in INVARIANT_KEYS:
        if base["config"].get(k) != quant["config"].get(k):
            raise ValueError("口径不一致：%s 基线=%r 量化=%r"
                             % (k, base["config"].get(k), quant["config"].get(k)))
    bp, qp = base["perf"], quant["perf"]
    res = {"base": base["name"], "quant": quant["name"], "perf": {}, "accuracy": {}}
    res["perf"]["ttft_p50_ratio"] = round(qp["ttft_p50"] / bp["ttft_p50"], 3)    # <1 更好
    res["perf"]["ttft_p99_ratio"] = round(qp["ttft_p99"] / bp["ttft_p99"], 3)    # <1 更好
    res["perf"]["tpot_p50_ratio"] = round(qp["tpot_p50"] / bp["tpot_p50"], 3)    # <1 更好
    res["perf"]["output_throughput_ratio"] = round(
        qp["output_throughput"] / bp["output_throughput"], 3)                    # >1 更好
    for t, v in base["accuracy"].items():
        if t in quant["accuracy"]:
            res["accuracy"][t] = {"base": v, "quant": quant["accuracy"][t],
                                  "delta_pp": round((quant["accuracy"][t] - v) * 100, 2)}
    for tag, r in (("base", base), ("quant", quant)):
        roof = roofline_tok_s(r["param_count_b"], r["weight_bits"], r["mem_bandwidth_gb_s"])
        decode = 1000.0 / r["perf"]["tpot_p50"]          # 解码速度 tok/s
        res["roofline_" + tag] = {"roofline_tok_s": round(roof, 2),
                                  "decode_tok_s": round(decode, 2),
                                  "achieved": round(decode / roof, 3)}
    return res


# ====================================================================
# 主流程
# ====================================================================
def run_once(cfg):
    os.makedirs(RESULT_DIR, exist_ok=True)
    name = cfg["name"]

    wait_for_server(cfg["base_url"], cfg["model"])

    perf_path = os.path.join(RESULT_DIR, "perf_%s.json" % name)
    subprocess.run(build_bench_cmd(cfg), check=True)
    perf = parse_bench_result(perf_path)
    perf["peak_memory_gb"] = peak_memory_gb()

    subprocess.run(build_eval_cmd(cfg), check=True)
    hits = sorted(glob.glob(os.path.join(RESULT_DIR, "eval_%s*.json" % name)),
                  key=os.path.getmtime)
    if not hits:
        sys.exit("没有找到 lm_eval 的输出文件 %s/eval_%s*.json" % (RESULT_DIR, name))
    acc = parse_eval_result(hits[-1])

    agent = None
    trace = cfg.get("agent_trace")
    if trace and os.path.exists(trace):
        agent = agent_metrics(trace)
    elif trace:
        # 前面的性能和精度结果已经跑完了，不能因为 trace 还没生成就让整次运行白费
        print("提示：没有找到 %s，本次不统计 Agent 指标" % trace)

    record = {
        "name": name,
        "model": cfg["model"],
        "config": {k: cfg[k] for k in INVARIANT_KEYS},
        "weight_bits": cfg["weight_bits"],
        "param_count_b": cfg["param_count_b"],
        "mem_bandwidth_gb_s": cfg["mem_bandwidth_gb_s"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "perf": perf,
        "accuracy": acc,
        "agent": agent,
    }
    out = os.path.join(RESULT_DIR, "run_%s.json" % name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print("写入", out)
    print(json.dumps({"perf": perf, "accuracy": acc, "agent": agent},
                     ensure_ascii=False, indent=2))
    return record


# ====================================================================
# 自测：不需要 GPU，也不需要服务在跑
# ====================================================================
def _fixture(name, bits, ttft, tpot, thr, acc):
    return {
        "name": name, "model": "m-" + name, "weight_bits": bits,
        "param_count_b": 8.03, "mem_bandwidth_gb_s": 273.0,
        "config": {k: DEFAULT_CONFIG[k] for k in INVARIANT_KEYS},
        "perf": {"ttft_p50": ttft, "ttft_p99": ttft * 2,
                 "tpot_p50": tpot, "output_throughput": thr,
                 "peak_memory_gb": 20.0},
        "accuracy": acc, "agent": None,
    }


def _numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)


def selftest():
    import tempfile
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + label + ("" if cond else "  " + detail))
        ok = ok and cond

    # --- 屋顶线 ---
    r16 = roofline_tok_s(8.03, 16, 273.0)
    r8 = roofline_tok_s(8.03, 8, 273.0)
    check("屋顶线：BF16 约 17 tok/s", abs(r16 - 17.0) < 0.5, "得到 %.2f" % r16)
    check("屋顶线：比特数减半，上限翻倍", abs(r8 / r16 - 2.0) < 1e-6)

    # --- 命令拼装：只有模型相关字段可以不同 ---
    base_cfg = dict(DEFAULT_CONFIG, name="base", model="qwen3-8b-bf16", weight_bits=16)
    fp8_cfg = dict(DEFAULT_CONFIG, name="fp8", model="qwen3-8b-fp8", weight_bits=8)
    cmd = " ".join(build_bench_cmd(fp8_cfg))
    check("压测命令包含模型名", "qwen3-8b-fp8" in cmd)
    check("压测命令包含并发设置", "--max-concurrency %d" % fp8_cfg["concurrency"] in cmd)
    check("压测命令包含输入输出长度",
          str(fp8_cfg["input_len"]) in cmd and str(fp8_cfg["output_len"]) in cmd)
    check("压测命令固定输出长度（--ignore-eos）", "--ignore-eos" in cmd)
    check("压测命令写出结果文件", "perf_fp8.json" in cmd)
    ecmd = " ".join(build_eval_cmd(fp8_cfg))
    check("评测命令包含任务列表", fp8_cfg["eval_tasks"] in ecmd)
    check("评测命令包含样本数上限", "--limit %d" % fp8_cfg["eval_limit"] in ecmd)
    strip = lambda c: [x for x in c if "qwen3-8b" not in x and "base" not in x and "fp8" not in x]
    check("两次的压测命令除模型与文件名外完全一致",
          strip(build_bench_cmd(base_cfg)) == strip(build_bench_cmd(fp8_cfg)))

    # --- 结果解析 ---
    with tempfile.TemporaryDirectory() as td:
        pf = os.path.join(td, "perf.json")
        json.dump({"median_ttft_ms": 100.0, "p99_ttft_ms": 120.0,
                   "median_tpot_ms": 20.0, "output_throughput": 48.0}, open(pf, "w"))
        pr = parse_bench_result(pf)
        check("解析压测结果", pr["tpot_p50"] == 20.0 and pr["output_throughput"] == 48.0)
        json.dump({"median_ttft_ms": 100.0}, open(pf, "w"))
        try:
            parse_bench_result(pf)
            check("缺字段时报错而不是返回 0", False)
        except KeyError:
            check("缺字段时报错而不是返回 0", True)
        ef = os.path.join(td, "eval.json")
        json.dump({"results": {"gsm8k": {"exact_match,strict-match": 0.9},
                               "arc_challenge": {"acc,none": 0.6, "acc_norm,none": 0.55}}},
                  open(ef, "w"))
        er = parse_eval_result(ef)
        check("解析评测结果（主指标）", er == {"gsm8k": 0.9, "arc_challenge": 0.55}, str(er))

        # --- Agent 指标 ---
        tf = os.path.join(td, "trace.jsonl")
        with open(tf, "w", encoding="utf-8") as f:
            f.write(json.dumps({"steps": [1, 2], "success": True, "final_output": "{\"a\": 1}",
                                "tool_calls": [{"args_valid": True, "error": None},
                                               {"args_valid": True, "error": "boom"}]}) + "\n")
            f.write(json.dumps({"steps": [1], "success": False, "final_output": "not json",
                                "tool_calls": [{"args_valid": False, "error": None}]}) + "\n")
        am = agent_metrics(tf)
        check("工具调用成功要求参数合法且执行无异常", am["tool_call_ok_rate"] == round(1 / 3, 4), str(am))
        check("结构化输出合法率", am["schema_ok_rate"] == 0.5, str(am))
        check("任务完成率与平均步数", am["task_success_rate"] == 0.5 and am["avg_steps"] == 1.5)
        empty = os.path.join(td, "empty.jsonl")
        open(empty, "w").close()
        em = agent_metrics(empty)
        check("没有数据时不编造 0%", em["tool_call_ok_rate"] is None and em["avg_steps"] is None)
        try:
            agent_metrics(os.path.join(td, "missing.jsonl"))
            check("没有 trace 文件时给出明确错误", False)
        except FileNotFoundError:
            check("没有 trace 文件时给出明确错误", True)

    # --- 对比：同口径 ---
    base = _fixture("base", 16, 900.0, 70.0, 13.6, {"gsm8k": 0.82})
    quant = _fixture("fp8", 8, 850.0, 41.0, 23.2, {"gsm8k": 0.815})
    res = compare(base, quant)
    check("对比产出了吞吐倍数 1.7", abs(res["perf"]["output_throughput_ratio"] - 23.2 / 13.6) < 0.01,
          str(res["perf"]))
    check("对比产出了精度差值（百分点）", abs(res["accuracy"]["gsm8k"]["delta_pp"] + 0.5) < 1e-6)
    check("对比产出了屋顶线达成率", all(0.5 <= res[k]["achieved"] <= 1.0
                                       for k in ("roofline_base", "roofline_quant")), str(res))

    # --- 对比：不同口径必须被挡住 ---
    bad = _fixture("fp8", 8, 850.0, 41.0, 23.2, {"gsm8k": 0.815})
    bad["config"]["concurrency"] = DEFAULT_CONFIG["concurrency"] + 7
    try:
        compare(base, bad)
        check("并发数不一致时报错", False, "compare 应当抛异常")
    except ValueError as e:
        check("并发数不一致时报错并指出字段", "concurrency" in str(e), str(e))

    print()
    print("全部通过" if ok else "存在未通过项，请检查输出")
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(
        description="量化前后的同口径对比：健康检查 → 性能压测 → 精度评测 → Agent 指标")
    ap.add_argument("--selftest", action="store_true", help="不需要 GPU 的自测")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "QUANT"),
                    help="对比两次运行的 results/run_*.json")
    d = DEFAULT_CONFIG
    ap.add_argument("--model", help="服务的 --served-model-name（必填）")
    ap.add_argument("--weight-bits", type=int, help="权重比特数，BF16 填 16，FP8 填 8（必填）")
    ap.add_argument("--name", default=d["name"], help="结果文件名前缀，默认「模型名_c并发数」")
    ap.add_argument("--tokenizer", default=d["tokenizer"], help="分词器目录，默认 %(default)s")
    ap.add_argument("--base-url", default=d["base_url"], help="服务地址，默认用 $PORT：%(default)s")
    ap.add_argument("--param-count-b", type=float, default=d["param_count_b"],
                    help="参数量（十亿），默认 %(default)s")
    ap.add_argument("--bandwidth", type=float, default=d["mem_bandwidth_gb_s"],
                    help="内存带宽 GB/s，默认 %(default)s（DGX Spark）")
    ap.add_argument("--concurrency", type=int, default=d["concurrency"], help="并发数，默认 %(default)s")
    ap.add_argument("--input-len", type=int, default=d["input_len"], help="输入长度，默认 %(default)s")
    ap.add_argument("--output-len", type=int, default=d["output_len"], help="输出长度，默认 %(default)s")
    ap.add_argument("--num-prompts", type=int, default=d["num_prompts"], help="请求数，默认 %(default)s")
    ap.add_argument("--temperature", type=float, default=d["temperature"], help="默认 %(default)s")
    ap.add_argument("--seed", type=int, default=d["seed"], help="默认 %(default)s")
    ap.add_argument("--eval-tasks", default=d["eval_tasks"], help="lm_eval 任务，默认 %(default)s")
    ap.add_argument("--eval-limit", type=int, default=d["eval_limit"],
                    help="每个任务的样本数，默认 %(default)s")
    ap.add_argument("--num-fewshot", type=int, default=d["num_fewshot"], help="默认 %(default)s")
    ap.add_argument("--agent-trace", default=d["agent_trace"],
                    help="Agent trace 文件（jsonl），留空则不统计 Agent 指标")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.compare:
        a, b = (json.load(open(p, encoding="utf-8")) for p in args.compare)
        print(json.dumps(compare(a, b), ensure_ascii=False, indent=2))
        return
    if not args.model or not args.weight_bits:
        ap.error("需要 --model 和 --weight-bits，或者用 --selftest / --compare")

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "model": args.model, "weight_bits": args.weight_bits, "tokenizer": args.tokenizer,
        "base_url": args.base_url, "param_count_b": args.param_count_b,
        "mem_bandwidth_gb_s": args.bandwidth, "concurrency": args.concurrency,
        "input_len": args.input_len, "output_len": args.output_len,
        "num_prompts": args.num_prompts, "temperature": args.temperature, "seed": args.seed,
        "eval_tasks": args.eval_tasks, "eval_limit": args.eval_limit,
        "num_fewshot": args.num_fewshot, "agent_trace": args.agent_trace,
    })
    cfg["name"] = args.name or "%s_c%d" % (args.model, args.concurrency)
    run_once(cfg)


if __name__ == "__main__":
    main()
