"""Terminal-Bench tasks run by OpenAI Agents SDK or LangGraph agents through
the Xybernetex adapters, on Workers AI - the Python-framework twin of the
OpenClaw survey (scenarios/tbench.py), on the same tasks and graders.

The agent runs on this machine; its four tools (run a command, read, write,
list) act inside the task's own container, where it may go anywhere - the
container is the sandbox. Grading uses the task's own verifier:

  first try   graded right after the first turn, on a `docker commit` of the
              container (graders install packages and copy tests in, so they
              never run where the agent still works)
  final       graded after any follow-ups (a check turn, contract fixes)

--contract auto adds the determinism layer: the run's model writes acceptance
checks while the agent works; they run in a scratch container from a
snapshot of the current state; a failed contract gets --max-fixes targeted
fix turns under the ratchet (snapshot = docker commit; restore = carry on in
a fresh container of the snapshot - background processes don't survive it).

    $env:PYTHONPATH = "..\\xybernetex-python\\src"; $env:CF_ACCOUNT_ID = ...; $env:CF_WORKERS_AI_KEY = ...
    python -m scenarios.tb_py --framework langgraph --model glm53f --from-survey --batch lg-tb
    python -m scenarios.tb_py --framework sdk --model glm53f --from-survey --survey-failed --contract auto --max-fixes 2 --batch sdk-tb-contract

Records go to scenarios/batches/tb-py/<batch>.jsonl; rerunning a batch
resumes it. Task images are dropped after each task. Terminal-Bench tasks
carry a canary: we score on them and never publish their content.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message=".*(Pydantic V1|create_react_agent has been moved).*")

from scenarios.run_suite import MODELS
from scenarios.tbench import SURVEY_DIR, Container, Task, drop_images, grade, grade_image, sh, surveyed, task_message

OUT_DIR = SURVEY_DIR.parent / "tb-py"
OUTPUT_LIMIT = 20_000
INSTRUCTIONS = (
    "You are an autonomous agent working in a Linux container. Use the tools to inspect files and run commands; "
    "don't just describe what you would do. Don't ask questions - make reasonable assumptions and finish the task "
    "completely, then reply with a short summary of what you did."
)


def clip(text: str) -> str:
    return text if len(text) <= OUTPUT_LIMIT else text[:OUTPUT_LIMIT] + f"\n...[{len(text) - OUTPUT_LIMIT} more characters]"


class Box:
    """The agent's container, which the ratchet can swap for a fresh one of a snapshot.
    Every image and container it makes is removed on close()."""

    def __init__(self, task: Task, image: str, wd: str):
        self.task, self.wd = task, wd
        self.c = Container(image, task.memory_mb, wd)
        self.gen = 0                  # bumped by anything that may change the container
        self._images: list[str] = []
        self._check: tuple[int, Container] | None = None

    # -- the agent's tools ------------------------------------------------------------------
    def run(self, command: str, timeout: int) -> tuple[int, str]:
        self.gen += 1
        try:
            p = subprocess.run(["docker", "exec", "-w", self.wd, self.c.name, "timeout", str(timeout), "bash", "-lc", command],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return 124, ""
        return p.returncode, p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")

    def write(self, path: str, content: str) -> None:
        self.gen += 1
        self.c.put(content, path if path.startswith("/") else f"{self.wd.rstrip('/')}/{path}")

    # -- snapshots: grading, contract checks and the ratchet ----------------------------------
    def commit(self) -> str:
        tag = f"xyb-tbpy-snap:{uuid.uuid4().hex[:12]}"
        sh(["docker", "commit", self.c.name, tag], timeout=1800)
        self._images.append(tag)
        return tag

    def check(self, command: str, timeout: int) -> tuple[int, str]:
        """A contract check, in a scratch container of the current state (one per state)."""
        if self._check is None or self._check[0] != self.gen:
            if self._check is not None:
                self._check[1].remove()
            self._check = (self.gen, Container(self.commit(), self.task.memory_mb, self.wd))
        scratch = self._check[1]
        try:
            p = subprocess.run(["docker", "exec", "-w", self.wd, scratch.name, "timeout", str(timeout), "bash", "-lc", command],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return 124, ""
        return p.returncode, p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")

    def snapshot(self) -> str:
        return self.commit()

    def restore(self, token: str) -> None:
        old, self.c = self.c, Container(token, self.task.memory_mb, self.wd)
        old.remove()
        self.gen += 1

    def discard(self, token: str) -> None:
        pass  # images go at close(); one may still back the current container

    def close(self) -> None:
        if self._check is not None:
            self._check[1].remove()
        self.c.remove()
        for tag in self._images:
            sh(["docker", "rmi", "-f", tag], check=False)


def tool_functions(box: Box) -> list:
    """The four tools as plain async functions; each framework wraps them its own way."""

    async def run_command(command: str, timeout_seconds: int = 120) -> str:
        """Run a bash command in the container (from the task's working directory) and return its output and exit status.

        Args:
            command: The command line to run.
            timeout_seconds: Kill the command after this many seconds (at most 1800).
        """
        limit = max(1, min(int(timeout_seconds), 1800))
        code, out = await asyncio.to_thread(box.run, command, limit)
        if code == 124:
            out += f"\ntimed out after {limit}s"
        return clip(out.rstrip() + f"\n[exit {code}]")

    async def read_file(path: str) -> str:
        """Read a text file in the container.

        Args:
            path: An absolute path, or one relative to the task's working directory.
        """
        code, out = await asyncio.to_thread(box.run, f"cat -- {shlex.quote(path)}", 60)
        return clip(out) if code == 0 else f"error: {out.strip()[-300:]}"

    async def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file in the container.

        Args:
            path: An absolute path, or one relative to the task's working directory.
            content: The whole file's new content.
        """
        try:
            await asyncio.to_thread(box.write, path, content)
            return f"wrote {len(content)} characters to {path}"
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"[:300]

    async def list_files(path: str = ".") -> str:
        """List files under a directory in the container (three levels deep).

        Args:
            path: The directory, absolute or relative to the task's working directory.
        """
        code, out = await asyncio.to_thread(
            box.run, f"find {shlex.quote(path)} -maxdepth 3 -not -path '*/.git/*' -printf '%y %s %p\\n' | head -400", 60)
        return clip(out) or "(empty)"

    return [run_command, read_file, write_file, list_files]


def record_line(r: dict) -> str:
    if "crash" in r:
        return f"{r['task']:<44} CRASH {r['crash'][:90]}"
    f, v, c = r["first"], r.get("verify"), r.get("contract")
    return (f"{r['task']:<44} first {f['reward']!s:<5} {f['status']:<7} {(f.get('usage') or {}).get('total', 0):>8} tok "
            f"${f['dollars']:.4f}" + (f"   final {r['reward_final']!s:<5}" if v or c else "")
            + (f"   follow-up ${v['dollars']:.4f}" if v else "")
            + (f"   contract {'met' if c and c['met'] else 'none' if c and not c['available'] else 'FAILED'} {c['ratchet']}"
               if c else "") + f"   {r['seconds']:.0f}s")


async def run_task(rel: str, args, sem: asyncio.Semaphore) -> dict:
    from scenarios.sdk_agent import contract_record, dollars, gate_summary
    async with sem:
        task = Task(rel)
        out = {"task": rel, "framework": args.framework, "model": args.model, "batch": args.batch,
               "at": datetime.now(timezone.utc).isoformat()}
        t0 = time.time()
        box = None
        try:
            image = await asyncio.to_thread(task.base_image)
            wd = await asyncio.to_thread(task.workdir, image)
            box = await asyncio.to_thread(Box, task, image, wd)
            from xybernetex.core.followups import RunSummary  # noqa: F401 - fail early if the adapter is missing
            events: list[dict] = []
            first: dict = {}

            async def grade_first(rep) -> None:
                # The first try alone, graded on a snapshot before anything else runs.
                first["reward"] = (await asyncio.to_thread(grade_image, task, await asyncio.to_thread(box.commit), wd))["reward"]
                first["graded_in_turn"] = True

            fw = FRAMEWORKS[args.framework]
            report = None
            rounds = args.max_fixes if args.contract else (1 if args.verify_model else 0)
            try:
                report = await asyncio.wait_for(fw.run(box, task, wd, args, events, grade_first),
                                                timeout=args.turn_timeout * (1 + rounds))
            except asyncio.TimeoutError:
                out["timeout"] = True
            out["seconds"] = round(time.time() - t0, 1)
            if "reward" not in first:  # timed out in the first turn: what it left is the first try
                first["reward"] = (await asyncio.to_thread(grade_image, task, await asyncio.to_thread(box.commit), wd))["reward"]
            usage = fw.usage(report.result) if report and report.result is not None else None
            out["first"] = {"status": "timeout" if report is None else report.status,
                            "error": None if report is None else report.error, "usage": usage,
                            "dollars": round(dollars(usage, MODELS[args.model][0]), 5), "reward": first["reward"],
                            "decision": report.decision if report else None, "governor": report.governor if report else None,
                            "reply": ((fw.reply(report.result) if report and report.result is not None else "") or "")[-600:],
                            "gate": gate_summary(events)}
            followed = report is not None and report.followup is not None
            if followed:
                follow_model = args.verify_model or args.model
                usage2 = fw.usage(report.followup)
                out["verify"] = {"model": follow_model, "action": (report.decision or {}).get("rule"), "usage": usage2,
                                 "dollars": round(dollars(usage2, MODELS[follow_model][0]), 5),
                                 "reply": (fw.reply(report.followup) or "")[-600:]}
            if args.contract and report is not None:
                out["contract"] = contract_record(report, args.model)
            # A follow-up (or a run cut off after its first turn was graded) may have changed things since.
            changed_since = followed or (out.get("timeout") and first.get("graded_in_turn"))
            out["reward_final"] = (await asyncio.to_thread(grade, task, box.c))["reward"] if changed_since else first["reward"]
        except Exception as e:  # noqa: BLE001 - one task's crash is a result, not the end of the batch
            out["seconds"] = round(time.time() - t0, 1)
            out["crash"] = f"{type(e).__name__}: {e}"[:300]
            traceback.print_exc()
        finally:
            if box is not None:
                await asyncio.to_thread(box.close)
            await asyncio.to_thread(drop_images, task)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / f"{args.batch}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(out, default=str) + "\n")
        print(record_line(out), flush=True)
        return out


# -- the two frameworks ---------------------------------------------------------------------

def _credentials() -> tuple[str, str]:
    account, key = os.environ.get("CF_ACCOUNT_ID"), os.environ.get("CF_WORKERS_AI_KEY")
    if not account or not key:
        sys.exit("set CF_ACCOUNT_ID and CF_WORKERS_AI_KEY in the environment (never in this file)")
    return f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1", key


class SdkFramework:
    def __init__(self):
        from agents import set_tracing_disabled
        from openai import AsyncOpenAI
        set_tracing_disabled(True)
        base, key = _credentials()
        self.client = AsyncOpenAI(base_url=base, api_key=key, max_retries=10)

    def model(self, short: str):
        from agents import OpenAIChatCompletionsModel
        return OpenAIChatCompletionsModel(model=MODELS[short][0], openai_client=self.client)

    async def run(self, box: Box, task: Task, wd: str, args, events: list, after_first):
        from agents import Agent, function_tool
        from xybernetex.openai_agents import Xybernetex
        from scenarios.sdk_agent import contract_settings
        xyb = Xybernetex(mode="enforce", preset="recommended", approvals=False, log=events.append,
                         followups={"mode": "act" if args.verify_model or args.contract else "observe"},
                                 governor=args.governor)
        agent = Agent(name="worker", instructions=INSTRUCTIONS, model=self.model(args.model),
                      tools=[function_tool(f) for f in tool_functions(box)])
        return await xyb.run(agent, task_message(task, wd), session_key=f"tb-{uuid.uuid4().hex[:8]}", max_turns=args.max_turns,
                             followup_model=self.model(args.verify_model) if args.verify_model else None,
                             contract=args.contract, run_checks=box.check if args.contract else None,
                             snapshots=box if args.contract and not args.no_ratchet else None,
                             max_fixes=args.max_fixes, after_first=after_first,
                             contract_settings=contract_settings() if args.contract else None)

    @staticmethod
    def usage(result):
        from scenarios.sdk_agent import usage_of
        return usage_of(result)

    @staticmethod
    def reply(result):
        return result.final_output if isinstance(result.final_output, str) else str(result.final_output or "")


class LangGraphFramework:
    def __init__(self):
        _credentials()

    @staticmethod
    def model(short: str, **settings):
        from scenarios.lg_agent import chat_model
        return chat_model(short, **settings)

    async def run(self, box: Box, task: Task, wd: str, args, events: list, after_first):
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent
        from xybernetex.langgraph import Xybernetex
        xyb = Xybernetex(mode="enforce", preset="recommended", approvals=False, log=events.append,
                         followups={"mode": "act" if args.verify_model or args.contract else "observe"},
                                 governor=args.governor)
        tools = [tool(f) for f in tool_functions(box)]
        graph = create_react_agent(self.model(args.model), tools=xyb.tool_node(tools), prompt=INSTRUCTIONS)
        follow = (create_react_agent(self.model(args.verify_model), tools=xyb.tool_node(tools), prompt=INSTRUCTIONS)
                  if args.verify_model else None)
        config = {"configurable": {"thread_id": f"tb-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 2 * args.max_turns + 1}
        return await xyb.run(graph, task_message(task, wd), config=config, followup_graph=follow,
                             model=MODELS[args.model][0], contract=args.contract,
                             contract_model=self.model(args.model, reasoning_effort="low") if args.contract else None,
                             run_checks=box.check if args.contract else None,
                             snapshots=box if args.contract and not args.no_ratchet else None,
                             max_fixes=args.max_fixes, after_first=after_first)

    @staticmethod
    def usage(result):
        from scenarios.lg_agent import usage_of
        return usage_of(result)

    @staticmethod
    def reply(result):
        from scenarios.lg_agent import reply_of
        return reply_of(result)


FRAMEWORKS: dict = {}


def summary(rows: list[dict]) -> str:
    from scenarios.sdk_agent import unusable
    ran = [r for r in rows if not unusable(r)]
    n = len(ran)
    first = sum(1 for r in ran if r["first"]["reward"] == 1.0)
    final = sum(1 for r in ran if r.get("reward_final") == 1.0)
    cost = sum(r["first"]["dollars"] + (r.get("verify") or {}).get("dollars", 0) + (r.get("contract") or {}).get("dollars", 0)
               for r in ran)
    return (f"\n{n} tasks: first try {first}/{n} ({first / max(n, 1):.0%}), final {final}/{n} ({final / max(n, 1):.0%}), "
            f"${cost:.4f} total = ${cost / max(final, 1):.4f} per solved task"
            + (f"; crashed or rate-limited: {len(rows) - n} (rerun the same command to retry them)" if len(rows) > n else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--framework", choices=["sdk", "langgraph"], required=True)
    parser.add_argument("--model", default="glm53f", help=f"short name: {', '.join(MODELS)}")
    parser.add_argument("--verify-model", help="a check-your-work turn on this model after the first turn")
    parser.add_argument("--contract", choices=["auto"], help="write a contract, check it, fix only what fails")
    parser.add_argument("--governor", choices=["standard"], help="stop runs that spend without progress (core/governor.py)")
    parser.add_argument("--max-fixes", type=int, default=2)
    parser.add_argument("--no-ratchet", action="store_true")
    parser.add_argument("--tasks", help="comma list of task paths, e.g. archive/fix-git")
    parser.add_argument("--from-survey", action="store_true", help="every task the OpenClaw survey validated (74)")
    parser.add_argument("--survey-failed", action="store_true", help="with --from-survey: only those OpenClaw didn't solve")
    parser.add_argument("--limit", type=int, help="at most this many tasks")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--turn-timeout", type=int, default=1380, help="seconds per turn, as in the OpenClaw survey")
    parser.add_argument("--max-turns", type=int, default=80, help="model/tool rounds per turn")
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    elif args.from_survey:
        tasks = surveyed(valid_only=True)
        if args.survey_failed:
            rows = [json.loads(line) for line in (SURVEY_DIR / "survey.jsonl").read_text(encoding="utf-8").splitlines()]
            solved = {r["task"] for r in rows if r.get("reward") == 1.0}
            tasks = [t for t in tasks if t not in solved]
    else:
        parser.error("--tasks or --from-survey is required")
    if args.limit:
        tasks = tasks[:args.limit]
    for m in (args.model, args.verify_model):
        if m and m not in MODELS:
            sys.exit(f"unknown model {m!r}; known: {', '.join(MODELS)}")
    FRAMEWORKS[args.framework] = (SdkFramework if args.framework == "sdk" else LangGraphFramework)()

    from scenarios.sdk_agent import unusable
    log = OUT_DIR / f"{args.batch}.jsonl"
    done = [r for r in map(json.loads, log.read_text(encoding="utf-8").splitlines())] if log.exists() else []
    done = [r for r in done if not unusable(r)]
    todo = [t for t in tasks if t not in {r["task"] for r in done}]
    print(f"{len(todo)} task(s) to run on {args.framework}/{args.model}"
          + (f", check on {args.verify_model}" if args.verify_model else "")
          + (f", contract with up to {args.max_fixes} fixes{'' if not args.no_ratchet else ' (no ratchet)'}" if args.contract else "")
          + f" ({len(done)} done), {args.concurrency} at a time", flush=True)

    async def go():
        sem = asyncio.Semaphore(max(1, args.concurrency))
        return await asyncio.gather(*(run_task(t, args, sem) for t in todo))
    rows = done + list(asyncio.run(go()))
    if rows:
        print(summary(rows), flush=True)
    print(f"\nlog: {log}")


if __name__ == "__main__":
    main()
