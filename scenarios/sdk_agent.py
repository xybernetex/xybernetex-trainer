"""One of our tasks, run by an OpenAI Agents SDK agent on a Workers AI model,
graded by our own grader. The first step of the Python port: the same tasks,
models and grading as the OpenClaw experiments, so results compare directly.

    pip install openai-agents
    set CF_ACCOUNT_ID=...          (PowerShell: $env:CF_ACCOUNT_ID = "...")
    set CF_WORKERS_AI_KEY=...
    python -m scenarios.sdk_agent --task hh-toposort --model glm53f
    python -m scenarios.sdk_agent --task hh-toposort --model glm53f --verify-model dsv4p

The SDK talks to Workers AI through its OpenAI-compatible endpoint (the same
one OpenClaw uses), so a run costs what our experiments cost, not GPT prices.
Chat completions, not the Responses API: Workers AI implements only the former.
The agent gets the four tools an OpenClaw coding agent has (run a command,
read, write, list), confined to a fresh folder seeded with the task's files.
--verify-model runs a check-your-work turn on a second model in the same
conversation - the "stronger model does the follow-up" lever - and grades
the folder again. The key is read from the environment and never printed.
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
    from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
    from agents.exceptions import MaxTurnsExceeded
    from openai import AsyncOpenAI
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing package ({e.name}): pip install openai-agents")

from scenarios.outcomes import evaluate
from scenarios.run_suite import FOLLOW_UPS, MODELS
from scenarios.suite import SCENARIOS

RUNS = Path.home() / ".xybernetex-sdk"
INSTRUCTIONS = (
    "This is an automated test task. Work only inside the current folder - create, change and delete files "
    "only there. Use the tools to look at the files and to run commands; don't just describe what you would do. "
    "Don't ask questions or introduce yourself; make reasonable assumptions and finish the task, then reply with "
    "a short summary of what you did."
)
OUTPUT_LIMIT = 20_000  # characters of tool output returned to the model


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
    folder = RUNS / f"{scenario['id']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
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


async def turn(agent: Agent, task_input, max_turns: int) -> tuple[object | None, str, float]:
    """One run of the agent: (result or None, status, seconds)."""
    t0 = time.time()
    try:
        result = await Runner.run(agent, task_input, max_turns=max_turns)
        return result, "done", round(time.time() - t0, 1)
    except MaxTurnsExceeded:
        return None, "max-turns", round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001 - the run's death is the result
        return None, f"failed: {type(e).__name__}: {str(e)[:200]}", round(time.time() - t0, 1)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", default="hh-toposort", help="scenario id (scenarios/suite.py)")
    parser.add_argument("--model", default="glm53f", help=f"short name: {', '.join(MODELS)}")
    parser.add_argument("--verify-model", help="run a check-your-work turn on this model afterwards")
    parser.add_argument("--max-turns", type=int, default=40, help="tool-call rounds before the SDK stops the run")
    parser.add_argument("--xybernetex", action="store_true",
                        help="run through the xybernetex-python adapter (gate, deaths, follow-ups) instead of bare Runner.run; "
                             "needs that package importable, e.g. PYTHONPATH=../xybernetex-python/src")
    args = parser.parse_args()
    scenario = next((s for s in SCENARIOS if s["id"] == args.task), None)
    if scenario is None:
        sys.exit(f"unknown task {args.task!r}")
    for m in (args.model, args.verify_model):
        if m and m not in MODELS:
            sys.exit(f"unknown model {m!r}; known: {', '.join(MODELS)}")

    client = client_from_env()
    set_tracing_disabled(True)  # tracing would go to OpenAI, with the wrong key
    folder = seed(scenario)
    tools = make_tools(folder)
    model = lambda short: OpenAIChatCompletionsModel(model=MODELS[short][0], openai_client=client)  # noqa: E731
    out = {"task": scenario["id"], "model": args.model, "folder": str(folder)}
    agent = Agent(name="worker", instructions=INSTRUCTIONS, model=model(args.model), tools=tools)

    if args.xybernetex:
        # The real product path: the gate on every tool call, death detection, and the
        # follow-up decided and run by the adapter (xybernetex-python), graded here.
        from xybernetex.openai_agents import Xybernetex
        events = []
        xyb = Xybernetex(mode="enforce", preset="recommended", approvals=False, log=events.append,
                         followups={"mode": "act" if args.verify_model else "observe"})
        report = await xyb.run(agent, scenario["prompt"], session_key=folder.name, max_turns=args.max_turns,
                               followup_model=model(args.verify_model) if args.verify_model else None)
        reply = report.result.final_output if report.result else None
        grade = evaluate(scenario, folder, reply)
        out["first"] = {"status": report.status, "error": report.error, "seconds": report.seconds,
                        "usage": usage_of(report.result) if report.result else None, "passed": grade.task,
                        "detail": grade.detail, "reply": (reply or "")[-600:], "decision": report.decision,
                        "gate": [e for e in events if e["type"].startswith("tool_gate") and e["type"] != "tool_gate_ready"]}
        print(json.dumps(out["first"], indent=1, default=str), flush=True)
        if report.followup is not None:
            reply2 = report.followup.final_output
            grade2 = evaluate(scenario, folder, reply2)
            out["verify"] = {"model": args.verify_model, "usage": usage_of(report.followup), "passed": grade2.task,
                             "detail": grade2.detail, "reply": (reply2 or "")[-600:]}
            print(json.dumps(out["verify"], indent=1, default=str), flush=True)
    else:
        result, status, seconds = await turn(agent, scenario["prompt"], args.max_turns)
        reply = result.final_output if result else None
        grade = evaluate(scenario, folder, reply)
        out["first"] = {"status": status, "seconds": seconds, "usage": usage_of(result) if result else None,
                        "passed": grade.task, "detail": grade.detail, "reply": (reply or "")[-600:]}
        print(json.dumps(out["first"], indent=1), flush=True)

        if args.verify_model and result:
            checker = Agent(name="checker", instructions=INSTRUCTIONS, model=model(args.verify_model), tools=tools)
            history = result.to_input_list() + [{"role": "user", "content": FOLLOW_UPS["verify"]}]
            result2, status2, seconds2 = await turn(checker, history, args.max_turns)
            reply2 = result2.final_output if result2 else reply
            grade2 = evaluate(scenario, folder, reply2)
            out["verify"] = {"model": args.verify_model, "status": status2, "seconds": seconds2,
                             "usage": usage_of(result2) if result2 else None, "passed": grade2.task,
                             "detail": grade2.detail, "reply": (reply2 or "")[-600:]}
            print(json.dumps(out["verify"], indent=1), flush=True)

    log = RUNS / "runs.jsonl"
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({**out, "at": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(f"\nfolder: {folder}\nlog: {log}")


if __name__ == "__main__":
    asyncio.run(main())
