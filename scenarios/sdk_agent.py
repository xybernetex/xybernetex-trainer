"""Our tasks, run by OpenAI Agents SDK agents on Workers AI models through the
Xybernetex Python adapter, graded by our own grader. The same tasks, models,
prompts, sandbox and grading as the OpenClaw experiments, so the numbers compare.

    pip install openai-agents
    $env:PYTHONPATH = "..\\xybernetex-python\\src"     (the adapter lives in xybernetex-python)
    $env:CF_ACCOUNT_ID = "..."; $env:CF_WORKERS_AI_KEY = "..."

    python -m scenarios.sdk_agent --task hh-toposort --model glm53f --verify-model dsv4p
    python -m scenarios.sdk_agent --category hard2 --model glm53f --verify-model dsv4p --batch sdk-hard2

Per task: a fresh folder seeded with the task's files; an agent with the four
tools an OpenClaw coding agent has (run a command, read, write, list),
confined to that folder, with commands run in the same Linux sandbox image and
network as OpenClaw's scenario agent (the folder mounted at /workspace);
Xybernetex.run with the gate in enforce mode (recommended preset, no
approvals: holds are explained blocks); the folder graded; then, with
--verify-model, the adapter's check-your-work turn on that model in the same
conversation, and the folder graded again. Every run is appended to
~/.xybernetex-sdk/runs.jsonl; a category run ends with a table. With
--batch NAME, runs are tagged and tasks already recorded under that name are
skipped, so an interrupted batch picks up where it stopped. One task failing
(an API error, a crash) is recorded and never stops the others.

The SDK talks to Workers AI through its OpenAI-compatible chat-completions
endpoint (the Responses API isn't implemented there). The key is read from
the environment and never printed. Costs use Workers AI list prices
(2026-09-28), with cached input at the cached rate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import traceback
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

from scenarios.outcomes import IMAGE, evaluate
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
NETWORK = "bridge"  # as OpenClaw's scenarios agent (sandbox.docker.network)
# USD per million tokens: input, cached input, output
# (developers.cloudflare.com/workers-ai/platform/pricing, 2026-09-28)
PRICE = {
    "@cf/zai-org/glm-5.3-flash": (0.15, 0.03, 0.50), "@cf/zai-org/glm-5.3": (1.40, 0.26, 4.40),
    "@cf/zai-org/glm-4.7-flash": (0.06, 0.06, 0.40), "@cf/deepseek-ai/deepseek-v4-flash-0731": (0.44, 0.014, 1.32),
    "@cf/deepseek-ai/deepseek-v4-pro-0813": (1.32, 0.044, 3.96), "@cf/qwen/qwen3.8-27b": (0.45, 0.45, 3.20),
    "@cf/nvidia/nemotron-3-120b-a12b": (0.50, 0.50, 1.50), "@cf/google/gemma-4-26b-a4b-it": (0.10, 0.10, 0.30),
    "@cf/moonshotai/kimi-k2.7-code": (0.95, 0.19, 4.00),
}


# Contracts are written at low reasoning effort: on detailed requests GLM-5.3 Flash and DeepSeek V4 Flash
# deliberated past 8,000+ tokens (or Workers AI's request limit) and wrote nothing; at low effort GLM wrote
# a contract in ~500 tokens.
def contract_settings():
    from agents import ModelSettings
    from agents.model_settings import Reasoning
    return ModelSettings(reasoning=Reasoning(effort="low"))


def fail_fast_on_timeouts() -> None:
    """Don't retry a 408. The OpenAI client retries a request timeout like a rate
    limit, but on Workers AI a 408 means the model deliberated past the ~4-minute
    request limit, and a retry deliberates again: in lg-hard3-check two tasks spent
    ~44 minutes each on 10 retries before dying anyway. Failing fast hands the dead
    run to the follow-up (a retry, maybe on a stronger model) right away. 429s and
    5xx still retry. Covers LangChain's ChatOpenAI too, which uses this client.
    (Client-side timeouts are still retried; both runners' timeouts are longer than
    Workers AI's, so its 408 arrives first.)"""
    from openai._base_client import BaseClient
    if getattr(BaseClient._should_retry, "_xybernetex", False):
        return
    original = BaseClient._should_retry

    def should_retry(self, response) -> bool:
        return False if response.status_code == 408 else original(self, response)

    should_retry._xybernetex = True
    BaseClient._should_retry = should_retry


def client_from_env() -> AsyncOpenAI:
    account, key = os.environ.get("CF_ACCOUNT_ID"), os.environ.get("CF_WORKERS_AI_KEY")
    if not account or not key:
        sys.exit("set CF_ACCOUNT_ID and CF_WORKERS_AI_KEY in the environment (never in this file)")
    # Workers AI rate-limits requests per minute per model (429): the client waits
    # (honoring retry-after) and retries rather than failing the run. A 408 doesn't retry.
    fail_fast_on_timeouts()
    return AsyncOpenAI(base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1", api_key=key,
                       max_retries=10)


# Failures of our setup, not of the task: a rate limit or a lost connection
# says nothing about the model or the framework. Such runs are left out of the
# summary and re-run by the next --batch invocation, like crashes.
INFRA_ERRORS = ("RateLimitError", "Error code: 429", "rate limiting", "APIConnectionError")


def unusable(r: dict) -> bool:
    error = str(((r.get("first") or {}).get("error")) or "")
    return "crash" in r or any(e in error for e in INFRA_ERRORS)


class Sandbox:
    """One container per task: OpenClaw's scenario sandbox image, the task folder
    mounted read-write at /workspace, kept alive for the run's commands."""

    def __init__(self, folder: Path):
        self.id = subprocess.run(
            ["docker", "run", "-d", "--rm", "--network", NETWORK, "-v", f"{folder.resolve()}:/workspace",
             "-w", "/workspace", "-e", "PYTHONDONTWRITEBYTECODE=1", IMAGE, "sleep", "infinity"],
            capture_output=True, text=True, check=True).stdout.strip()

    def run(self, command: str, timeout: int) -> tuple[int, str]:
        try:
            p = subprocess.run(["docker", "exec", "-w", "/workspace", self.id, "timeout", str(timeout), "bash", "-c", command],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return 124, ""
        return p.returncode, p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")

    def check(self, command: str, timeout: int) -> tuple[int, str]:
        """A contract check, run on a throwaway copy of the folder so it can't change the work."""
        setup = subprocess.run(["docker", "exec", "-w", "/", self.id, "bash", "-c",
                                "rm -rf /tmp/xyb-check && cp -a /workspace /tmp/xyb-check"],
                               capture_output=True, text=True, timeout=120)
        if setup.returncode != 0:
            raise RuntimeError(f"couldn't copy the folder for checking: {setup.stderr[-200:]}")
        try:
            p = subprocess.run(["docker", "exec", "-w", "/tmp/xyb-check", self.id, "timeout", str(timeout), "bash", "-c",
                                command], capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return 124, ""
        return p.returncode, p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")

    # The ratchet's snapshots (xybernetex core/ratchet.py): full copies of the folder inside the container.
    _snaps = 0

    def _sh(self, script: str) -> None:
        p = subprocess.run(["docker", "exec", "-w", "/", self.id, "bash", "-c", script], capture_output=True, text=True,
                           timeout=300)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-300:] or f"exit {p.returncode}")

    def snapshot(self) -> int:
        self._snaps += 1
        self._sh(f"rm -rf /tmp/xyb-snap-{self._snaps} && cp -a /workspace /tmp/xyb-snap-{self._snaps}")
        return self._snaps

    def restore(self, token: int) -> None:
        # cp -r, not -a: the folder is a host mount, where preserving times on /workspace itself fails.
        self._sh(f"find /workspace -mindepth 1 -delete && cp -r /tmp/xyb-snap-{int(token)}/. /workspace/")

    def discard(self, token: int) -> None:
        self._sh(f"rm -rf /tmp/xyb-snap-{int(token)}")

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.id], capture_output=True)


def make_tools(folder: Path, sandbox: Sandbox) -> list:
    """The agent's tools, confined to `folder`. Paths are relative to it."""

    def inside(rel: str) -> Path:
        rel = rel.removeprefix("/workspace/").removeprefix("/workspace")
        p = (folder / rel).resolve()
        if p != folder.resolve() and folder.resolve() not in p.parents:
            raise ValueError(f"{rel!r} is outside the task folder")
        return p

    def clip(text: str) -> str:
        return text if len(text) <= OUTPUT_LIMIT else text[:OUTPUT_LIMIT] + f"\n...[{len(text) - OUTPUT_LIMIT} more characters]"

    @function_tool
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
    cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
    return {"requests": u.requests, "input": u.input_tokens, "cached": cached, "output": u.output_tokens,
            "total": u.total_tokens}


def dollars(usage: dict | None, model_id: str) -> float:
    if not usage or model_id not in PRICE:
        return 0.0
    pi, pc, po = PRICE[model_id]
    cached = min(usage.get("cached", 0), usage["input"])
    return ((usage["input"] - cached) * pi + cached * pc + usage["output"] * po) / 1e6


def gate_summary(events: list[dict]) -> list[dict]:
    return [{k: e[k] for k in ("type", "action", "authorization", "waiver", "riskTier", "toolName") if k in e}
            for e in events if e["type"].startswith("tool_gate") and e["type"] != "tool_gate_ready"]


async def run_task(scenario: dict, args, client: AsyncOpenAI, sem: asyncio.Semaphore) -> dict:
    """One task through the adapter: first turn, grade, optional follow-up, grade again."""
    model = lambda short: OpenAIChatCompletionsModel(model=MODELS[short][0], openai_client=client)  # noqa: E731
    async with sem:
        folder = seed(scenario)
        out = {"task": scenario["id"], "model": args.model, "batch": args.batch, "folder": str(folder),
               "at": datetime.now(timezone.utc).isoformat()}
        t0 = time.time()
        try:
            sandbox = await asyncio.to_thread(Sandbox, folder)
            try:
                agent = Agent(name="worker", instructions=INSTRUCTIONS, model=model(args.model),
                              tools=make_tools(folder, sandbox))
                events: list[dict] = []
                xyb = Xybernetex(mode="enforce", preset="recommended", approvals=False, log=events.append,
                                 followups={"mode": "act" if args.verify_model or args.contract else "observe"},
                                 governor=args.governor)
                first: dict = {}

                async def grade_first(rep):
                    # Graded here, before any check or follow-up can change the folder.
                    first["reply"] = rep.result.final_output if rep.result else None
                    first["grade"] = await asyncio.to_thread(evaluate, scenario, folder, first["reply"])
                report = await xyb.run(agent, scenario["prompt"], session_key=folder.name, max_turns=args.max_turns,
                                       followup_model=model(args.verify_model) if args.verify_model else None,
                                       contract=args.contract, run_checks=sandbox.check if args.contract else None,
                                       snapshots=sandbox if args.contract and not args.no_ratchet else None,
                                       max_fixes=args.max_fixes, after_first=grade_first,
                                       contract_settings=contract_settings() if args.contract and args.writer_effort == "low"
                                       else None,
                                       contract_version=args.contract_version,
                                       judge_model=model(args.verify_model or args.model)
                                       if args.contract and args.contract_version == "v2" else None)
            finally:
                await asyncio.to_thread(sandbox.close)
            out["seconds"] = round(time.time() - t0, 1)
            reply, grade = first["reply"], first["grade"]
            usage = usage_of(report.result) if report.result else None
            out["first"] = {"status": report.status, "error": report.error, "seconds": report.seconds, "usage": usage,
                            "dollars": round(dollars(usage, MODELS[args.model][0]), 5), "passed": grade.task,
                            "detail": grade.detail, "reply": (reply or "")[-600:], "decision": report.decision, "governor": report.governor,
                            "gate": gate_summary(events)}
            if report.followup is not None:
                reply2 = report.followup.final_output
                grade2 = await asyncio.to_thread(evaluate, scenario, folder, reply2)
                usage2 = usage_of(report.followup)
                follow_model = args.verify_model or args.model
                out["verify"] = {"model": follow_model, "action": (report.decision or {}).get("rule"), "usage": usage2,
                                 "passed": grade2.task, "detail": grade2.detail,
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


def contract_record(report, model_short: str, judge_short: str | None = None) -> dict:
    """What the contract arm did: the contract, every check round, and what writing it (and, in v2,
    judging it) cost."""
    c = report.contract
    tokens = report.contract_tokens or 0
    _, _, out_price = PRICE.get(MODELS[model_short][0], (0, 0, 0))
    judge_tokens = getattr(report, "judge_tokens", 0) or 0
    _, _, judge_price = PRICE.get(MODELS[judge_short or model_short][0], (0, 0, 0))
    return {"available": c is not None, "source": c.source if c else None, "checks": len(c.checks) if c else 0,
            "refused": len(c.refused) if c else 0, "hash": c.hash if c else None, "tokens": tokens,
            # Usage for the one writing call isn't split into input and output: priced at the
            # output rate, an upper bound.
            "dollars": round(tokens * out_price / 1e6 + judge_tokens * judge_price / 1e6, 5),
            "rounds": [v.summary() for v in report.verdicts], "ratchet": list(report.ratchet), "met": report.contract_met,
            "judgments": list(getattr(report, "judgments", [])), "judge_tokens": judge_tokens,
            "overturned": getattr(c, "overturned", 0) if c else 0}


def line_for(r: dict) -> str:
    if "crash" in r or unusable(r):
        if "crash" not in r:
            return f"{r['task']:<18} INFRA {r['first']['error'][:100]}"
        return f"{r['task']:<18} CRASH {r['crash'][:100]}"
    f1, v = r["first"], r.get("verify")
    return (f"{r['task']:<18} first {'PASS' if f1['passed'] else 'fail':<4} {f1['status']:<6} "
            f"{(f1['usage'] or {}).get('total', 0):>7} tok ${f1['dollars']:.4f}"
            + (f"   verify {'PASS' if v['passed'] else 'fail':<4} {v['usage']['total']:>7} tok ${v['dollars']:.4f}" if v else "")
            + (f"   gate: {', '.join(e.get('action') or e['type'] for e in f1['gate'])}" if f1["gate"] else "")
            + (f"   contract: {'none' if not r['contract']['available'] else 'met' if r['contract']['met'] else 'FAILED'}"
               if r.get("contract") else "")
            + f"   {r['seconds']:.0f}s")


def summary(results: list[dict], verify: bool) -> str:
    ran = [r for r in results if not unusable(r)]
    n = len(ran)
    first_pass = sum(1 for r in ran if r["first"]["passed"])
    first_cost = sum(r["first"]["dollars"] for r in ran)
    line = f"\n{n} tasks: first turn {first_pass}/{n} passed ({first_pass / max(n, 1):.0%}), ${first_cost:.4f}"
    if verify:
        after = sum(1 for r in ran if (r.get("verify") or r["first"])["passed"])
        total = first_cost + sum((r.get("verify") or {}).get("dollars", 0) + (r.get("contract") or {}).get("dollars", 0)
                                 for r in ran)
        line += (f"; after the check {after}/{n} ({after / max(n, 1):.0%}), ${total:.4f} total"
                 f" = ${total / max(after, 1):.4f} per completed task (first turn alone: ${first_cost / max(first_pass, 1):.4f})")
    blocked = sum(1 for r in ran for e in r["first"]["gate"] if e.get("action") == "BLOCK_ACTION")
    crashed = len(results) - n
    return line + f"; gate blocks: {blocked}" + (f"; crashed or rate-limited: {crashed} (rerun the same command to retry them)"
                                                   if crashed else "")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", help="one scenario id (scenarios/suite.py)")
    parser.add_argument("--category", help="every scenario in this category, e.g. hard2 or hard3")
    parser.add_argument("--model", default="glm53f", help=f"short name: {', '.join(MODELS)}")
    parser.add_argument("--verify-model", help="the adapter runs a check-your-work turn on this model afterwards")
    parser.add_argument("--max-turns", type=int, default=40, help="tool-call rounds before the SDK stops a run")
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

    client = client_from_env()
    set_tracing_disabled(True)  # tracing would go to OpenAI, with the wrong key
    sem = asyncio.Semaphore(max(1, args.concurrency))
    print(f"{len(chosen)} task(s) on {args.model}" + (f", check on {args.verify_model}" if args.verify_model else "")
          + f", {args.concurrency} at a time", flush=True)
    results = done + list(await asyncio.gather(*(run_task(s, args, client, sem) for s in chosen)))
    if len(results) > 1:
        print(summary(results, bool(args.verify_model or args.contract)), flush=True)
    print(f"\nlog: {RUNS / 'runs.jsonl'}")


if __name__ == "__main__":
    asyncio.run(main())
