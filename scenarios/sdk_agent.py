"""Our tasks, run by OpenAI Agents SDK agents on Workers AI models through the
Xybernetex Python adapter, graded by our own grader. The same tasks, models,
prompts and grading as the OpenClaw experiments, so the numbers compare.

    pip install openai-agents
    $env:PYTHONPATH = "..\\xybernetex-python\\src"     (the adapter lives in xybernetex-python)
    $env:CF_ACCOUNT_ID = "..."; $env:CF_WORKERS_AI_KEY = "..."

    python -m scenarios.sdk_agent --task hh-toposort --model glm53f --verify-model dsv4p
    python -m scenarios.sdk_agent --category hard2 --model glm53f --verify-model dsv4p --concurrency 2

Per task: a fresh folder seeded with the task's files; an agent with the four
tools an OpenClaw coding agent has (run a command, read, write, list),
confined to that folder; Xybernetex.run with the gate in enforce mode
(recommended preset, no approvals: holds are explained blocks); the folder
graded; then, with --verify-model, the adapter's check-your-work turn on that
model in the same conversation, and the folder graded again. Every run is
appended to ~/.xybernetex-sdk/runs.jsonl; a category run ends with a table.

The SDK talks to Workers AI through its OpenAI-compatible chat-completions
endpoint (the Responses API isn't implemented there). The key is read from
the environment and never printed. Costs use Workers AI list prices
(2026-09-28); the SDK's usage doesn't separate cached input, so they are
upper bounds for models that cache.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from agents import Agent, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
    from openai import AsyncOpenAI
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing package ({e.name}): pip install openai-agents")
try:
    from xybernetex.openai_agents import Xybernetex
except ImportError:  # pragma: no cover
    sys.exit("the adapter isn't importable: set PYTHONPATH to ..\\xybernetex-python\\src (or pip install -e it)")

from scenarios.outcomes import evaluate
from scenarios.run_suite import MODELS
from scenarios.suite import SCENARIOS

RUNS = Path.home() / ".xybernetex-sdk"
INSTRUCTIONS = (
    "This is an automated test task. Work only inside the current folder - create, change and delete files "
    "only there. Use the tools to look at the files and to run commands; don't just describe what you would do. "
    "Don't ask questions or introduce yourself; make reasonable assumptions and finish the task, then reply with "
    "a short summary of what you did."
)
OUTPUT_LIMIT = 20_000  # characters of tool output returned to the model
# USD per million tokens: input, output (developers.cloudflare.com/workers-ai/platform/pricing, 2026-09-28)
PRICE = {
    "@cf/zai-org/glm-5.3-flash": (0.15, 0.50), "@cf/zai-org/glm-5.3": (1.40, 4.40), "@cf/zai-org/glm-4.7-flash": (0.06, 0.40),
    "@cf/deepseek-ai/deepseek-v4-flash-0731": (0.44, 1.32), "@cf/deepseek-ai/deepseek-v4-pro-0813": (1.32, 3.96),
    "@cf/qwen/qwen3.8-27b": (0.45, 3.20), "@cf/nvidia/nemotron-3-120b-a12b": (0.50, 1.50),
    "@cf/google/gemma-4-26b-a4b-it": (0.10, 0.30), "@cf/moonshotai/kimi-k2.7-code": (0.95, 4.00),
}


def client_from_env() -> AsyncOpenAI:
    account, key = os.environ.get("CF_ACCOUNT_ID"), os.environ.get("CF_WORKERS_AI_KEY")
    if not account or not key:
        sys.exit("set CF_ACCOUNT_ID and CF_WORKERS_AI_KEY in the environment (never in this file)")
    return AsyncOpenAI(base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1", api_key=key)


def make_tools(folder: Path) -> list:
    """The agent's tools, confined to `folder`. Paths are relative to it."""

    def inside(rel: str) -> Path:
        p = (folder / rel).resolve()
        if p != folder.resolve() and folder.resolve() not in p.parents:
            raise ValueError(f"{rel!r} is outside the task folder")
        return p

    def clip(text: str) -> str:
        return text if len(text) <= OUTPUT_LIMIT else text[:OUTPUT_LIMIT] + f"\n...[{len(text) - OUTPUT_LIMIT} more characters]"

    @function_tool
    def run_command(command: str, timeout_seconds: int = 60) -> str:
        """Run a shell command in the task folder and return its output and exit status.

        Args:
            command: The command line to run (Python is available as python3 and python).
            timeout_seconds: Kill the command after this many seconds.
        """
        try:
            p = subprocess.run(command, shell=True, cwd=folder, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=max(1, min(timeout_seconds, 600)))
        except subprocess.TimeoutExpired:
            return f"timed out after {timeout_seconds}s"
        out = p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")
        return clip(out.rstrip() + f"\n[exit {p.returncode}]")

    @function_tool
    def read_file(path: str) -> str:
        """Read a text file in the task folder.

        Args:
            path: The file's path, relative to the task folder.
        """
        try:
            return clip(inside(path).read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError) as e:
            return f"error: {e}"

    @function_tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file in the task folder.

        Args:
            path: The file's path, relative to the task folder.
            content: The whole file's new content.
        """
        try:
            p = inside(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            return f"wrote {len(content)} characters to {path}"
        except (OSError, ValueError) as e:
            return f"error: {e}"

    @function_tool
    def list_files() -> str:
        """List every file in the task folder, with sizes."""
        rows = [f"{p.relative_to(folder).as_posix()}  {p.stat().st_size} bytes"
                for p in sorted(folder.rglob("*")) if p.is_file()]
        return "\n".join(rows) or "(empty)"

    return [run_command, read_file, write_file, list_files]


def seed(scenario: dict) -> Path:
    folder = RUNS / f"{scenario['id']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    folder.mkdir(parents=True)
    for rel, content in (scenario.get("files") or {}).items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    return folder


def usage_of(result) -> dict:
    u = result.context_wrapper.usage
    return {"requests": u.requests, "input": u.input_tokens, "output": u.output_tokens, "total": u.total_tokens}


def dollars(usage: dict | None, model_id: str) -> float:
    if not usage or model_id not in PRICE:
        return 0.0
    pi, po = PRICE[model_id]
    return (usage["input"] * pi + usage["output"] * po) / 1e6


async def run_task(scenario: dict, args, client: AsyncOpenAI, sem: asyncio.Semaphore) -> dict:
    """One task through the adapter: first turn, grade, optional follow-up, grade again."""
    model = lambda short: OpenAIChatCompletionsModel(model=MODELS[short][0], openai_client=client)  # noqa: E731
    async with sem:
        folder = seed(scenario)
        agent = Agent(name="worker", instructions=INSTRUCTIONS, model=model(args.model), tools=make_tools(folder))
        events: list[dict] = []
        xyb = Xybernetex(mode="enforce", preset="recommended", approvals=False, log=events.append,
                         followups={"mode": "act" if args.verify_model else "observe"})
        t0 = time.time()
        report = await xyb.run(agent, scenario["prompt"], session_key=folder.name, max_turns=args.max_turns,
                               followup_model=model(args.verify_model) if args.verify_model else None)
        out = {"task": scenario["id"], "model": args.model, "folder": str(folder),
               "at": datetime.now(timezone.utc).isoformat(), "seconds": round(time.time() - t0, 1)}
        reply = report.result.final_output if report.result else None
        grade = evaluate(scenario, folder, reply)
        usage = usage_of(report.result) if report.result else None
        out["first"] = {"status": report.status, "error": report.error, "seconds": report.seconds, "usage": usage,
                        "dollars": round(dollars(usage, MODELS[args.model][0]), 5), "passed": grade.task,
                        "detail": grade.detail, "reply": (reply or "")[-600:], "decision": report.decision,
                        "gate": [{k: v for k, v in e.items() if k in ("type", "action", "authorization", "riskTier", "toolName")}
                                 for e in events if e["type"].startswith("tool_gate") and e["type"] != "tool_gate_ready"]}
        if report.followup is not None:
            reply2 = report.followup.final_output
            grade2 = evaluate(scenario, folder, reply2)
            usage2 = usage_of(report.followup)
            out["verify"] = {"model": args.verify_model, "usage": usage2, "passed": grade2.task, "detail": grade2.detail,
                             "dollars": round(dollars(usage2, MODELS[args.verify_model][0]), 5), "reply": (reply2 or "")[-600:]}
        RUNS.mkdir(parents=True, exist_ok=True)
        with open(RUNS / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(out, default=str) + "\n")
        f1, v = out["first"], out.get("verify")
        print(f"{scenario['id']:<18} first {'PASS' if f1['passed'] else 'fail':<4} {f1['status']:<6} "
              f"{(f1['usage'] or {}).get('total', 0):>7} tok ${f1['dollars']:.4f}"
              + (f"   verify {'PASS' if v['passed'] else 'fail':<4} {v['usage']['total']:>7} tok ${v['dollars']:.4f}" if v else "")
              + (f"   gate: {', '.join(e['action'] or e['type'] for e in f1['gate'])}" if f1["gate"] else ""), flush=True)
        return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", help="one scenario id (scenarios/suite.py)")
    parser.add_argument("--category", help="every scenario in this category, e.g. hard2 or hard3")
    parser.add_argument("--model", default="glm53f", help=f"short name: {', '.join(MODELS)}")
    parser.add_argument("--verify-model", help="the adapter runs a check-your-work turn on this model afterwards")
    parser.add_argument("--max-turns", type=int, default=40, help="tool-call rounds before the SDK stops a run")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.task:
        chosen = [s for s in SCENARIOS if s["id"] == args.task]
    elif args.category:
        chosen = [s for s in SCENARIOS if s["category"] == args.category]
    else:
        parser.error("--task or --category is required")
    if not chosen:
        sys.exit("no such task or category")
    for m in (args.model, args.verify_model):
        if m and m not in MODELS:
            sys.exit(f"unknown model {m!r}; known: {', '.join(MODELS)}")

    client = client_from_env()
    set_tracing_disabled(True)  # tracing would go to OpenAI, with the wrong key
    sem = asyncio.Semaphore(max(1, args.concurrency))
    print(f"{len(chosen)} task(s) on {args.model}" + (f", check on {args.verify_model}" if args.verify_model else "")
          + f", {args.concurrency} at a time", flush=True)
    results = await asyncio.gather(*(run_task(s, args, client, sem) for s in chosen))

    if len(results) > 1:
        n = len(results)
        first_pass = sum(1 for r in results if r["first"]["passed"])
        first_cost = sum(r["first"]["dollars"] for r in results)
        line = f"\n{n} tasks: first turn {first_pass}/{n} passed ({first_pass / n:.0%}), ${first_cost:.4f}"
        if args.verify_model:
            after = sum(1 for r in results if (r.get("verify") or r["first"])["passed"])
            total = first_cost + sum((r.get("verify") or {}).get("dollars", 0) for r in results)
            line += (f"; after the check {after}/{n} ({after / n:.0%}), ${total:.4f} total"
                     f" = ${total / max(after, 1):.4f} per completed task (first turn alone: ${first_cost / max(first_pass, 1):.4f})")
        blocked = sum(1 for r in results for e in r["first"]["gate"] if e["action"] == "BLOCK_ACTION")
        print(line + f"; gate blocks: {blocked}", flush=True)
    print(f"\nlog: {RUNS / 'runs.jsonl'}")


if __name__ == "__main__":
    asyncio.run(main())
