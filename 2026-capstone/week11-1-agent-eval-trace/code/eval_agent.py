"""在冻结任务集上运行 baseline / Planning，保存逐任务评分、trace 和汇总。"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from agent_v2 import (ROOT, MEMORY, ProjectTools, add_runtime_arguments, make_client,
                      run_agent, run_with_args, utc_now, validate_plan)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tasks(path):
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if not tasks:
        raise ValueError("任务集为空")
    seen = set()
    for task in tasks:
        task_id = task.get("task_id", "")
        if not isinstance(task_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", task_id) or task_id in seen:
            raise ValueError("task_id 需要唯一，并只包含英文字母、数字、短横线和下划线")
        seen.add(task_id)
        if not isinstance(task.get("input"), str) or not task["input"].strip():
            raise ValueError(f"{task_id} 缺少 input")
        if not isinstance(task.get("category"), str) or not task["category"]:
            raise ValueError(f"{task_id} 缺少 category")
        expected = task["expected"]
        if not isinstance(expected, dict) or not expected.get("answer_all"):
            raise ValueError(f"{task_id} 至少需要一个 answer_all 评分条件")
        for pattern in expected["answer_all"] + expected.get("answer_none", []):
            re.compile(pattern)
        for name, patterns in expected.get("artifacts", {}).items():
            if Path(name).is_absolute() or ".." in Path(name).parts or not name.startswith("notes/"):
                raise ValueError("评估产物需要位于 notes/ 下")
            for pattern in patterns:
                re.compile(pattern)
        if task.get("fixture", {}).get("fault") not in (None, "search_timeout_once"):
            raise ValueError("未知故障 fixture")
    return tasks


def grade(task, result, workspace):
    """代码评分器只在执行结束后读取 expected，不向 Agent 传递参考答案。"""
    expected = task["expected"]
    answer = result["answer"] or ""
    calls = [e for e in result["events"] if e["kind"] == "tool"]
    retrieved = {chunk["id"] for call in calls if call["name"] == "search_knowledge"
                 and call["error"] is None for chunk in call["observation"]}
    citations = set(result["citations"])
    checks = {"completed": result["status"] == "completed",
              "answer_content": all(re.search(p, answer, re.I) for p in expected["answer_all"]),
              "no_forbidden_content": not any(re.search(p, answer, re.I)
                                              for p in expected.get("answer_none", [])),
              "citations_retrieved": citations <= retrieved,
              "required_citations": all(citations & set(group)
                                        for group in expected.get("citation_groups", [])),
              "required_tools": set(expected.get("required_tools", [])) <= {
                  c["name"] for c in calls if isinstance(c["name"], str)}}
    for name, error_type in expected.get("required_reads", {}).items():
        checks[f"read_input:{name}"] = any(
            call["name"] == "read_file" and call.get("arguments") == {"path": name}
            and (call["error"] or {}).get("type") == error_type for call in calls)
    recovered = expected.get("recovered_tool")
    if recovered:
        relevant = [c for c in calls if c["name"] == recovered]
        errors = [i for i, c in enumerate(relevant) if c["error"] is not None]
        checks["tool_recovered"] = bool(errors) and any(c["error"] is None for c in relevant[errors[0] + 1:])
    artifacts = {}
    expected_files = expected.get("artifacts", {})
    notes = workspace / "notes"
    actual_files = {p.relative_to(workspace).as_posix() for p in notes.rglob("*") if p.is_file()}
    checks["artifact_files"] = actual_files == set(expected_files)
    for name, patterns in expected_files.items():
        path = workspace / name
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        checks[f"artifact_content:{name}"] = path.is_file() and all(re.search(p, text, re.I) for p in patterns)
    for name in sorted(actual_files):
        artifacts[name] = file_hash(workspace / name)
    return {"success": all(checks.values()), "checks": checks,
            "failed_checks": [key for key, passed in checks.items() if not passed],
            "artifact_sha256": artifacts}


def metrics(result):
    llm = [e for e in result["events"] if e["kind"] == "llm"]
    calls = [e for e in result["events"] if e["kind"] == "tool"]
    usages = [e.get("response", {}).get("usage") for e in llm]
    tokens = None
    if usages and all(isinstance(u, dict) and isinstance(u.get("total_tokens"), int) for u in usages):
        tokens = sum(u["total_tokens"] for u in usages)
    return {"latency_s": result["latency_s"], "llm_calls": result["llm_calls"],
            "tool_calls": len(calls), "tool_successes": sum(c["args_valid"] and c["error"] is None for c in calls),
            "replans": result["replans"], "total_tokens": tokens,
            "structured_final": result["status"] == "completed"}


def aggregate(rows):
    graded = [r for r in rows if r["evaluation_status"] == "ok"]
    tool_count = sum(r["metrics"]["tool_calls"] for r in rows)
    token_values = [r["metrics"]["total_tokens"] for r in rows]
    return {"trials": len(rows), "graded_trials": len(graded),
            "evaluation_errors": len(rows) - len(graded),
            "passed": sum(r["score"]["success"] for r in graded),
            "task_success_rate": sum(r["score"]["success"] for r in graded) / len(graded) if graded else None,
            "structured_final_rate": statistics.mean(r["metrics"]["structured_final"] for r in rows),
            "tool_success_rate": sum(r["metrics"]["tool_successes"] for r in rows) / tool_count if tool_count else None,
            "mean_latency_s": statistics.mean(r["metrics"]["latency_s"] for r in rows),
            "mean_llm_calls": statistics.mean(r["metrics"]["llm_calls"] for r in rows),
            "mean_tool_calls": statistics.mean(r["metrics"]["tool_calls"] for r in rows),
            "mean_replans": statistics.mean(r["metrics"]["replans"] for r in rows),
            "mean_total_tokens": statistics.mean(token_values) if all(t is not None for t in token_values) else None,
            "status_counts": dict(Counter(r["status"] for r in rows))}


def build_summary(rows, manifest, finished):
    return {"schema_version": 1, "finished": finished, "manifest": manifest,
            "by_mode": {mode: aggregate([r for r in rows if r["mode"] == mode])
                        for mode in sorted({r["mode"] for r in rows})},
            "by_category": {mode: {category: aggregate([r for r in rows if r["mode"] == mode
                                                       and r["category"] == category])
                                     for category in sorted({r["category"] for r in rows if r["mode"] == mode})}
                            for mode in sorted({r["mode"] for r in rows})}}


def regrade(source, tasks_path, output):
    """校验任务输入后，对两版本的全部原始 trace 统一重新评分。"""
    source = source.resolve()
    tasks = load_tasks(tasks_path)
    original_tasks = load_tasks(source / "tasks.jsonl")
    inputs = {t["task_id"]: {k: v for k, v in t.items() if k != "expected"} for t in tasks}
    original_inputs = {t["task_id"]: {k: v for k, v in t.items() if k != "expected"} for t in original_tasks}
    if inputs != original_inputs:
        raise ValueError("任务输入或 fixture 已发生变化，需要重新运行 Agent")
    original_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if not original_summary["finished"]:
        raise ValueError("请对已完成的整批评估重新评分")
    rows = [json.loads(line) for line in (source / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    output = output or source / ("regraded_" + utc_now().replace(":", "").replace("+", "_"))
    output.mkdir(parents=True, exist_ok=False)
    task_map = {t["task_id"]: t for t in tasks}
    manifest = {**original_summary["manifest"], "regraded_at": utc_now(),
                "original_results": str(source), "tasks_sha256": file_hash(tasks_path),
                "original_tasks_sha256": file_hash(source / "tasks.jsonl"),
                "code_sha256": {name: file_hash(ROOT / name) for name in ("agent_v2.py", "eval_agent.py")}}
    save_json(output / "manifest.json", manifest)
    shutil.copyfile(tasks_path, output / "tasks.jsonl")
    for row in rows:
        trace_path = source / row["trace"]
        result = json.loads(trace_path.read_text(encoding="utf-8"))
        workspace = source / "workspaces" / row["mode"] / f"{row['task_id']}_r{row['repeat']}"
        try:
            row.update(evaluation_status="ok", score=grade(task_map[row["task_id"]], result, workspace))
            row.pop("evaluation_error", None)
        except Exception as exc:
            row.update(evaluation_status="error", score=None,
                       evaluation_error=f"{type(exc).__name__}: {exc}")
        row["trace"] = Path(os.path.relpath(trace_path, output)).as_posix()
    with (output / "results.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = build_summary(rows, manifest, True)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary["by_mode"], ensure_ascii=False, indent=2))
    print(f"重新评分目录 {output.resolve()}")
    return 1 if any(r["evaluation_status"] == "error" for r in rows) else 0


def selftest(v1_dir, tasks_path):
    """确定性检查评估与控制逻辑，真实模型实验另存 results/。"""
    checks = 0

    def check(name, passed):
        nonlocal checks
        if not passed:
            raise AssertionError(name)
        checks += 1
        print(f"PASS {name}")

    def scripted(outputs):
        iterator = iter(outputs)

        def call(messages, event, remaining):
            return json.dumps(next(iterator), ensure_ascii=False)
        return call

    plan = {"steps": [{"id": "p1", "goal": "检索日志位置", "done_when": "获得资料片段"},
                      {"id": "p2", "goal": "回答", "done_when": "输出路径与来源"}]}
    search = {"action": "tool", "name": "search_knowledge", "arguments": {"query": "运行日志 路径"}}
    finish = {"action": "finish", "answer": "日志位于 logs/app.log。", "citations": ["logging.md#1"],
              "completed_steps": ["p1", "p2"]}
    tasks = load_tasks(tasks_path)
    check("任务集可读、编号唯一、评分正则有效", bool(tasks))
    check("端口设置接受等价中文表达", all(
        re.search(tasks[2]["expected"]["answer_all"][3], phrase)
        for phrase in ("APP_PORT=8001", "将 `APP_PORT` 为一个空闲端口，例如 8001", "APP_PORT 配置项为 8001")))
    with tempfile.TemporaryDirectory(prefix="week11-selftest-") as folder:
        workspace = Path(folder)
        tools = ProjectTools(v1_dir, workspace)
        check("复用第六周分块与词项检索", "logging.md#1" in {
            c["id"] for c in tools.execute("search_knowledge", {"query": "运行日志 路径"})})
        try:
            tools.execute("write_note", {"path": "../outside.md", "content": "x"})
        except ValueError:
            check("产物路径校验", True)
        else:
            check("产物路径校验", False)
        result = run_agent(tasks[0]["input"], "planning", scripted([plan, search, finish]), tools, MEMORY)
        check("Planner 到 Executor 正常入口", grade(tasks[0], result, workspace)["success"])
        check("模型输入只含任务与可见状态", "expected" not in result["events"][0]["state"])
        check("早期状态快照保持原样", not result["events"][0]["state"]["history"])
        baseline_finish = {**finish, "completed_steps": ["read_file"]}
        baseline = run_agent(tasks[0]["input"], "baseline", scripted([search, baseline_finish]), tools, MEMORY)
        check("baseline 使用同一工具与评分器", grade(tasks[0], baseline, workspace)["success"])
        check("关闭 Planning 时忽略计划进度字段", baseline["completed_steps"] == [])
        finish_without_progress = {"action": "finish", "answer": finish["answer"], "citations": finish["citations"]}
        partial_progress = run_agent(tasks[0]["input"], "planning",
                                     scripted([plan, search, finish_without_progress]), tools, MEMORY)
        check("最终结果保留待核验的计划进度", grade(tasks[0], partial_progress, workspace)["success"]
              and partial_progress["uncompleted_steps"] == ["p1", "p2"])
        faulty = ProjectTools(v1_dir, workspace, "search_timeout_once")
        recovered = run_agent(tasks[6]["input"], "planning",
                              scripted([plan, search, plan, search, finish]), faulty, MEMORY)
        check("一次超时触发重规划并恢复", grade(tasks[6], recovered, workspace)["success"]
              and recovered["replans"] == 1)
        missing_call = {"action": "tool", "name": "read_file", "arguments": {"path": "missing.txt"}}
        missing_finish = {"action": "finish", "answer": "missing.txt 文件不存在。", "citations": []}
        missing = run_agent(tasks[5]["input"], "baseline", scripted([missing_call, missing_finish]), tools, MEMORY)
        check("文件缺失由真实读取错误确认", grade(tasks[5], missing, workspace)["success"])
        wrong_read = json.loads(json.dumps(missing))
        wrong_read["events"][1]["arguments"]["path"] = "another.txt"
        check("读取其他文件无法满足指定输入", not grade(tasks[5], wrong_read, workspace)["success"])
        budget = run_agent("预算", "planning", scripted([plan]), tools, MEMORY, max_calls=1)
        check("模型调用预算可终止", budget["status"] == "call_budget")
        budget = run_agent("预算", "baseline", scripted([search]), tools, MEMORY, max_tools=0)
        check("工具预算可终止", budget["status"] == "tool_budget" and budget["tool_calls"] == 0)
        no_replan = run_agent("故障", "planning", scripted([plan, search]),
                              ProjectTools(v1_dir, workspace, "search_timeout_once"), MEMORY, max_replans=0)
        check("重规划预算可终止", no_replan["status"] == "replan_budget")
        malformed = run_agent("格式", "baseline", lambda *_: "not json", tools, MEMORY)
        check("无效输出保留模型原文", malformed["status"] == "error"
              and malformed["events"][0]["raw"] == "not json")
        timed = run_agent("超时", "baseline", scripted([]), tools, MEMORY, timeout=0)
        check("总时限可终止并标记 timeout", timed["status"] == "timeout")
        try:
            validate_plan({"steps": [plan["steps"][0], plan["steps"][0]]})
        except ValueError:
            check("重复计划 id 被识别", True)
        else:
            check("重复计划 id 被识别", False)
        fake = {**result, "answer": "已保存 notes/logging.md"}
        check("声称写入但文件缺失会失败", not grade(tasks[7], fake, workspace)["success"])
        tools.execute("write_note", {"path": "notes/logging.md", "content": "logs/app.log [logging.md#1]"})
        fake["events"] = result["events"] + [{"kind": "tool", "name": "write_note", "error": None}]
        check("评分读取真实产物", grade(tasks[7], fake, workspace)["success"])
        check("无写入任务发现额外产物", not grade(tasks[0], result, workspace)["success"])
        fake["citations"] = ["invented.md#1"]
        check("虚构引用不能通过", not grade(tasks[7], fake, workspace)["success"])
    print(f"{checks} checks passed. 真实模型结果需另外运行评估。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--regrade", type=Path, metavar="原始结果目录")
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks.jsonl")
    parser.add_argument("--mode", choices=["baseline", "planning", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if args.selftest:
        selftest(args.v1_dir, args.tasks)
        return 0
    if args.regrade:
        return regrade(args.regrade, args.tasks, args.output)
    for name in ("repeats", "max_tokens", "max_calls", "max_tools", "timeout", "request_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于零")
    if args.max_replans < 0:
        parser.error("--max-replans 不能小于零")
    tasks = load_tasks(args.tasks)
    # 提前检查资料。数据或环境错误发生在创建实验目录之前。
    v1 = ProjectTools(args.v1_dir, args.v1_dir / "workspace").v1
    client = make_client(args)
    client.healthcheck()
    output = args.output or ROOT / "results" / ("run_" + utc_now().replace(":", "").replace("+", "_"))
    output.mkdir(parents=True, exist_ok=False)
    modes = ["baseline", "planning"] if args.mode == "both" else [args.mode]
    source_paths = [args.v1_dir / "agent_v1.py", *sorted((args.v1_dir / "knowledge").rglob("*.md")),
                    *sorted((args.v1_dir / "workspace").rglob("*.txt"))]
    manifest = {"started_at": utc_now(), "python": platform.python_version(), "platform": platform.platform(),
                "tasks_sha256": file_hash(args.tasks), "task_count": len(tasks),
                "source_sha256": {p.relative_to(args.v1_dir).as_posix(): file_hash(p) for p in source_paths},
                "code_sha256": {name: file_hash(ROOT / name) for name in ("agent_v2.py", "eval_agent.py")},
                "memory": MEMORY, "modes": modes, "config": {k: str(v) if isinstance(v, Path) else v
                                                             for k, v in vars(args).items()},
                "seed_schedule": "seed + repeat - 1，两种模式使用相同 seed",
                "planned_trials": len(tasks) * args.repeats * len(modes)}
    save_json(output / "manifest.json", manifest)
    shutil.copyfile(args.tasks, output / "tasks.jsonl")
    warmup = {"kind": "llm", "phase": "warmup"}
    client([{"role": "user", "content": '输出 JSON {"ready":true}'}], warmup, args.request_timeout)
    save_json(output / "warmup.json", warmup)
    rows = []
    save_json(output / "summary.json", build_summary(rows, manifest, False))
    with (output / "results.jsonl").open("x", encoding="utf-8") as result_file:
        for repeat in range(1, args.repeats + 1):
            # 第一次 baseline 先跑，第二次 Planning 先跑，降低固定顺序造成的偏差。
            order = modes if repeat % 2 else list(reversed(modes))
            for task in tasks:
                for mode in order:
                    trial_id = f"{task['task_id']}_r{repeat}"
                    workspace = output / "workspaces" / mode / trial_id
                    shutil.copytree(args.v1_dir / "workspace", workspace)
                    save_json(workspace / "preferences.json", MEMORY)
                    notes = v1.load_memory(workspace / "preferences.json")
                    tools = ProjectTools(args.v1_dir, workspace, task.get("fixture", {}).get("fault"))
                    client.seed = args.seed + repeat - 1
                    result = run_with_args(task["input"], mode, client, tools, notes, args)
                    result.update(task_id=task["task_id"], repeat=repeat, seed=client.seed,
                                  dataset_sha256=manifest["tasks_sha256"])
                    trace_path = output / "traces" / mode / f"{trial_id}.json"
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    save_json(trace_path, result)
                    row = {"task_id": task["task_id"], "category": task["category"], "repeat": repeat,
                           "mode": mode, "seed": client.seed, "status": result["status"],
                           "stop_reason": result["stop_reason"], "metrics": metrics(result),
                           "trace": trace_path.relative_to(output).as_posix()}
                    try:
                        row.update(evaluation_status="ok", score=grade(task, result, workspace))
                    except Exception as exc:
                        row.update(evaluation_status="error", score=None,
                                   evaluation_error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
                    result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    result_file.flush()
                    save_json(output / "summary.json", build_summary(rows, manifest, False))
                    print(f"{mode:8} {trial_id} status={row['status']} score={row['score']}", flush=True)
    summary = build_summary(rows, manifest, True)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary["by_mode"], ensure_ascii=False, indent=2))
    print(f"结果目录 {output.resolve()}")
    return 1 if any(r["evaluation_status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
