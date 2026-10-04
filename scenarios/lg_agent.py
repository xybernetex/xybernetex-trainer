"""Our tasks, run by a LangGraph ReAct agent on Workers AI models through the
Xybernetex LangGraph adapter, graded by our own grader - the LangGraph twin of
scenarios/sdk_agent.py (same tasks, prompts, tools, sandbox, grading, prices
and output), so OpenClaw, the OpenAI Agents SDK and LangGraph compare.

    pip install langgraph langchain-openai
    $env:PYTHONPATH = "..\\xybernetex-python\\src"
    $env:CF_ACCOUNT_ID = "..."; $env:CF_WORKERS_AI_KEY = "..."

    python -m scenarios.lg_agent --task hh-toposort --model glm53f --verify-model dsv4p
    python -m scenarios.lg_agent --category hard2 --model glm53f --verify-model dsv4p --batch lg-hard2

Runs append to ~/.xybernetex-sdk/runs.jsonl with "framework": "langgraph";
--batch resumes an interrupted batch as in sdk_agent.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message=".*(Pydantic V1|create_react_agent has been moved).*")
try:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing package ({e.name}): pip install langgraph langchain-openai")
try:
    from xybernetex.langgraph import Xybernetex
except ImportError:  # pragma: no cover
    sys.exit("the adapter isn't importable: set PYTHONPATH to ..\\xybernetex-python\\src (or pip install -e it)")

from scenarios.outcomes import evaluate
from scenarios.run_suite import MODELS
from scenarios.sdk_agent import (INSTRUCTIONS, fail_fast_on_timeouts, OUTPUT_LIMIT, RUNS, Sandbox, contract_record, dollars, gate_summary,
                                 line_for, seed, summary, unusable)
from scenarios.suite import SCENARIOS


def chat_model(short: str, **settings) -> ChatOpenAI:
    account, key = os.environ.get("CF_ACCOUNT_ID"), os.environ.get("CF_WORKERS_AI_KEY")
    if not account or not key:
        sys.exit("set CF_ACCOUNT_ID and CF_WORKERS_AI_KEY in the environment (never in this file)")
    fail_fast_on_timeouts()   # a 408 is a death, not something to wait out
    return ChatOpenAI(model=MODELS[short][0], api_key=key, timeout=300, max_retries=10,  # 429s: wait and retry
                      base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1", **settings)


def make_tools(folder: Path, sandbox: Sandbox) -> list:
    """The same four tools as sdk_agent, as LangChain tools."""

    def inside(rel: str) -> Path:
        rel = rel.removeprefix("/workspace/").removeprefix("/workspace")
        p = (folder / rel).resolve()
        if p != folder.resolve() and folder.resolve() not in p.parents:
            raise ValueError(f"{rel!r} is outside the task folder")
        return p

    def clip(text: str) -> str:
        return text if len(text) <= OUTPUT_LIMIT else text[:OUTPUT_LIMIT] + f"\n...[{len(text) - OUTPUT_LIMIT} more characters]"

    @tool
    async def run_command(command: str, timeout_seconds: int = 60) -> str:
        """Run a bash command in the task folder (a Linux sandbox) and return its output and exit status.

        Args:
            command: The command line to run (Python is python3; git, node and curl are installed).
            timeout_seconds: Kill the command after this many seconds.
        """
        limit = max(1, min(timeout_seconds, 600))
        code, out = await asyncio.to_thread(sandbox.run, command, limit)
        if code == 124:
            out += f"\ntimed out after {limit}s"
        return clip(out.rstrip() + f"\n[exit {code}]")

    @tool
    def read_file(path: str) -> str:
        """Read a text file in the task folder.

        Args:
            path: The file's path, relative to the task folder.
        """
        try:
            return clip(inside(path).read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError) as e:
            return f"error: {e}"

    @tool
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

    @tool
    def list_files() -> str:
        """List every file in the task folder, with sizes."""
        rows = [f"{p.relative_to(folder).as_posix()}  {p.stat().st_size} bytes"
                for p in sorted(folder.rglob("*")) if p.is_file()]
        return "\n".join(rows) or "(empty)"

    return [run_command, read_file, write_file, list_files]


def turn_of(out: dict | None) -> list:
    """The messages after the latest human message: what this run added."""
    messages = list((out or {}).get("messages") or [])
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1:]
    return messages


def usage_of(out: dict | None) -> dict | None:
    ais = [m for m in turn_of(out) if isinstance(m, AIMessage)]
    if not ais:
        return None
    u = {"requests": len(ais), "input": 0, "cached": 0, "output": 0, "total": 0}
    for m in ais:
        meta = m.usage_metadata or {}
        u["input"] += int(meta.get("input_tokens") or 0)
        u["cached"] += int((meta.get("input_token_details") or {}).get("cache_read") or 0)
        u["output"] += int(meta.get("output_tokens") or 0)
        u["total"] += int(meta.get("total_tokens") or 0)
    return u


def reply_of(out: dict | None) -> str | None:
    final = next((m for m in reversed(turn_of(out)) if isinstance(m, AIMessage)), None)
    if final is None:
        return None
    return final.content if isinstance(final.content, str) else json.dumps(final.content)[:4000]


async def run_task(scenario: dict, args, sem: asyncio.Semaphore) -> dict:
    async with sem:
        folder = seed(scenario)
        out = {"task": scenario["id"], "model": args.model, "framework": "langgraph", "batch": args.batch,
               "folder": str(folder), "at": datetime.now(timezone.utc).isoformat()}
        t0 = time.time()
        try:
            sandbox = await asyncio.to_thread(Sandbox, folder)
            try:
                events: list[dict] = []
                xyb = Xybernetex(mode="enforce", preset="recommended", approvals=False, log=events.append,
                                 followups={"mode": "act" if args.verify_model or args.contract else "observe"},
                                 governor=args.governor)
                first: dict = {}

                async def grade_first(rep):
                    # Graded here, before any check or follow-up can change the folder.
                    first["reply"] = reply_of(rep.result)
                    first["grade"] = await asyncio.to_thread(evaluate, scenario, folder, first["reply"])
                tools = make_tools(folder, sandbox)
                graph = create_react_agent(chat_model(args.model), tools=xyb.tool_node(tools), prompt=INSTRUCTIONS)
                follow = (create_react_agent(chat_model(args.verify_model), tools=xyb.tool_node(tools), prompt=INSTRUCTIONS)
                          if args.verify_model else None)
                config = {"configurable": {"thread_id": folder.name}, "recursion_limit": 2 * args.max_turns + 1}
                report = await xyb.run(graph, scenario["prompt"], config=config, followup_graph=follow,
                                       model=MODELS[args.model][0], contract=args.contract,
                                       contract_model=(chat_model(args.model, **({"reasoning_effort": "low"}
                                                       if args.writer_effort == "low" else {})))
                                       if args.contract else None,  # see sdk_agent
                                       contract_version=args.contract_version,
                                       judge_model=chat_model(args.verify_model or args.model)
                                       if args.contract and args.contract_version == "v2" else None,
                                       run_checks=sandbox.check if args.contract else None,
                                       snapshots=sandbox if args.contract and not args.no_ratchet else None,
                                       max_fixes=args.max_fixes, after_first=grade_first)
            finally:
                await asyncio.to_thread(sandbox.close)
            out["seconds"] = round(time.time() - t0, 1)
            reply, grade = first["reply"], first["grade"]
            usage = usage_of(report.result)
            out["first"] = {"status": report.status, "error": report.error, "seconds": report.seconds, "usage": usage,
                            "dollars": round(dollars(usage, MODELS[args.model][0]), 5), "passed": grade.task,
                            "detail": grade.detail, "reply": (reply or "")[-600:], "decision": report.decision, "governor": report.governor,
                            "gate": gate_summary(events)}
            if report.followup is not None:
                reply2 = reply_of(report.followup)
                grade2 = await asyncio.to_thread(evaluate, scenario, folder, reply2)
                usage2 = usage_of(report.followup)
                follow_model = args.verify_model or args.model
                out["verify"] = {"model": follow_model, "action": (report.decision or {}).get("rule"),
                                 "usage": usage2, "passed": grade2.task, "detail": grade2.detail,
                                 "dollars": round(dollars(usage2, MODELS[follow_model][0]), 5),
                                 "reply": (reply2 or "")[-600:]}
            if args.contract:
                out["contract"] = contract_record(report, args.model, args.verify_model or args.model)
        except Exception as e:  # noqa: BLE001 - one task's crash is a result, not the end of the batch
            out["seconds"] = round(time.time() - t0, 1)
            out["crash"] = f"{type(e).__name__}: {e}"[:300]
            traceback.print_exc()
        RUNS.mkdir(parents=True, exist_ok=True)
        with open(RUNS / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(out, default=str) + "\n")
        print(line_for(out), flush=True)
        return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", help="one scenario id (scenarios/suite.py)")
    parser.add_argument("--category", help="every scenario in this category, e.g. hard2 or hard3")
    parser.add_argument("--model", default="glm53f", help=f"short name: {', '.join(MODELS)}")
    parser.add_argument("--verify-model", help="the adapter runs a follow-up turn on this model afterwards")
    parser.add_argument("--max-turns", type=int, default=40, help="model/tool rounds before LangGraph stops a run")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--batch", help="tag runs with this name and skip tasks already recorded under it")
    parser.add_argument("--contract", choices=["auto"], help="write a contract with --model, check it when the run "
                        "ends, and fix only what fails (on --verify-model if given, else the same model)")
    parser.add_argument("--governor", choices=["standard"], help="stop runs that spend without progress (core/governor.py)")
    parser.add_argument("--max-fixes", type=int, default=1, help="fix rounds for a failed contract")
    parser.add_argument("--no-ratchet", action="store_true", help="don't snapshot before fix turns or undo regressions")
    parser.add_argument("--contract-version", choices=["v1", "v2"], default="v1",
                        help="v2: checks must quote the request, a judge (--verify-model, else --model) rules on "
                             "failed checks before any fix, and the agent may dispute a check")
    parser.add_argument("--writer-effort", choices=["low", "normal"], default="low",
                        help="reasoning effort for the contract writer (low: GLM wrote nothing at normal effort on "
                             "detailed tasks; normal is the control for that choice)")
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
    chat_model(args.model)  # fail fast on missing credentials

    done: list[dict] = []
    if args.batch and (RUNS / "runs.jsonl").is_file():
        with open(RUNS / "runs.jsonl", encoding="utf-8") as f:
            done = [r for r in map(json.loads, f) if r.get("batch") == args.batch and not unusable(r)]
        finished = {r["task"] for r in done}
        chosen = [s for s in chosen if s["id"] not in finished]
        if done:
            print(f"batch {args.batch}: {len(done)} task(s) already done, {len(chosen)} to go", flush=True)
            for r in done:
                print(line_for(r))
    sem = asyncio.Semaphore(max(1, args.concurrency))
    print(f"{len(chosen)} task(s) on {args.model} (LangGraph)" + (f", follow-up on {args.verify_model}" if args.verify_model else "")
          + f", {args.concurrency} at a time", flush=True)
    results = done + list(await asyncio.gather(*(run_task(s, args, sem) for s in chosen)))
    if len(results) > 1:
        print(summary(results, bool(args.verify_model or args.contract)), flush=True)
    print(f"\nlog: {RUNS / 'runs.jsonl'}")


if __name__ == "__main__":
    asyncio.run(main())
