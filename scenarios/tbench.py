"""Terminal-Bench (Harbor-format) tasks in our counterfactual harness.

A Harbor task is a folder: instruction.md, task.toml, an environment (a
prebuilt docker_image or environment/Dockerfile), tests/test.sh (writes a
reward to /logs/verifier/reward.txt) and solution/solve.sh (a reference
solution). Tasks are self-contained, so this runs them without the Harbor
framework:

  1. image     the task's environment, plus Node and OpenClaw copied in from a
               shared runtime image (the task's own /usr/local is left alone)
  2. validate  the reference solution must pass the grader and the untouched
               environment must fail it, or the task is dropped
  3. run       OpenClaw runs inside a container of that image (embedded
               --local mode against Workers AI; the container is the sandbox,
               so exec runs without prompts)
  4. fork      after the first turn the container is committed: files AND
               OpenClaw's session state. Every follow-up branch starts from
               that snapshot in its own container, so even "continue the same
               session" is a true fork
  5. grade     each branch with the task's own tests

Provider credentials come from the host's OpenClaw config and are written only
into the container's config file, never printed. Terminal-Bench is Apache-2.0;
its tasks carry a canary asking that they never appear in training corpora:
we only score interventions on them and never publish task content.

    python -m scenarios.tbench validate --task archive/fix-git
    python -m scenarios.tbench prototype --task archive/fix-git --model glm53f
    python -m scenarios.tbench survey --model glm53f --concurrency 3
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import tempfile
import time
import tomllib
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

TBENCH = Path.home() / "tbench"
OPENCLAW_VERSION = "2026.9.6"
RUNTIME_IMAGE = f"xyb-openclaw-runtime:{OPENCLAW_VERSION}"
OC_WORKSPACE = "/root/oc-workspace"  # OpenClaw's own workspace, kept out of the task's directory
PROVIDER = "workers-ai"
SESSION = "tb"  # each container runs its own OpenClaw, so one session key serves every run
OC_STORE = "/root/.openclaw/agents/main/agent"
SURVEY_DIR = Path(__file__).parent / "batches" / "tb-survey"


def sh(args: list[str], timeout: int = 1800, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                       input=input_text)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:4])}... failed ({p.returncode}): {(p.stderr or p.stdout)[-600:]}")
    return p


class Task:
    def __init__(self, rel: str):
        self.rel, self.path = rel, TBENCH / rel
        self.cfg = tomllib.loads((self.path / "task.toml").read_text(encoding="utf-8"))
        env = self.cfg.get("environment", {})
        self.image = env.get("docker_image")
        self.name = self.path.name
        self.instruction = (self.path / "instruction.md").read_text(encoding="utf-8").strip()
        self.agent_timeout = int((self.cfg.get("agent") or {}).get("timeout_sec") or 900)
        self.verifier_timeout = int((self.cfg.get("verifier") or {}).get("timeout_sec") or 600)
        self.memory_mb = min(env.get("memory_mb") or 2048, 12288)  # a limit, not a reservation; Docker has 16 GB

    def base_image(self) -> str:
        """The task environment: pulled if prebuilt, else built from environment/."""
        if self.image:
            if sh(["docker", "image", "inspect", self.image], check=False).returncode != 0:
                sh(["docker", "pull", self.image], timeout=1800)
            return self.image
        tag = f"xyb-tb-env:{self.name}"
        if sh(["docker", "image", "inspect", tag], check=False).returncode != 0:
            sh(["docker", "build", "-t", tag, str(self.path / "environment")], timeout=3600)
        return tag

    def workdir(self, image: str) -> str:
        wd = (self.cfg.get("environment") or {}).get("workdir")
        return wd or sh(["docker", "image", "inspect", "-f", "{{.Config.WorkingDir}}", image]).stdout.strip() or "/"


def candidates(difficulties: tuple[str, ...] = ("easy", "medium")) -> list[str]:
    """Tasks this harness can run: Linux, CPU-only, single-step, with tests, a
    reference solution and an environment. Difficulty is the task's label, else
    its expert-time estimate (<=30 min easy, <=2 h medium)."""
    out, seen = [], set()
    for folder in ("tasks", "archive"):
        for toml in sorted((TBENCH / folder).glob("*/task.toml")):
            t = toml.parent
            cfg = tomllib.loads(toml.read_text(encoding="utf-8"))
            env, meta = cfg.get("environment", {}), cfg.get("metadata", {})
            if env.get("gpus") or env.get("gpu_types") or env.get("tpu") or env.get("os", "linux") != "linux":
                continue
            if (t / "steps").exists() or not (t / "tests" / "test.sh").exists() \
                    or not (t / "solution" / "solve.sh").exists():
                continue
            if not (t / "environment" / "Dockerfile").exists() and not env.get("docker_image"):
                continue
            hours = meta.get("expert_time_estimate_hours") or (meta.get("expert_time_estimate_min") or 0) / 60 or None
            diff = str(meta.get("difficulty") or "").lower() or (
                None if hours is None else "easy" if hours <= 0.5 else "medium" if hours <= 2 else "hard")
            if diff in difficulties and t.name not in seen:
                seen.add(t.name)
                out.append(f"{folder}/{t.name}")
    return out


def ensure_runtime() -> None:
    """Node + OpenClaw, installed once into a Debian image to copy from."""
    if sh(["docker", "image", "inspect", RUNTIME_IMAGE], check=False).returncode == 0:
        return
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Dockerfile").write_text(
            "FROM node:24-bookworm-slim\n"
            f"RUN npm install -g openclaw@{OPENCLAW_VERSION} && npm cache clean --force\n", encoding="utf-8")
        sh(["docker", "build", "-t", RUNTIME_IMAGE, d], timeout=3600)


def agent_image(task: Task) -> str:
    """The task environment with Node and OpenClaw added (only their own paths)."""
    base = task.base_image()
    ensure_runtime()
    tag = f"xyb-tb-agent:{task.name}"
    if sh(["docker", "image", "inspect", tag], check=False).returncode == 0:
        return tag
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Dockerfile").write_text(
            f"FROM {base}\n"
            f"COPY --from={RUNTIME_IMAGE} /usr/local/bin/node /usr/local/bin/node\n"
            f"COPY --from={RUNTIME_IMAGE} /usr/local/lib/node_modules/openclaw /opt/openclaw\n"
            "RUN ln -sf /opt/openclaw/openclaw.mjs /usr/local/bin/openclaw && chmod +x /opt/openclaw/openclaw.mjs\n",
            encoding="utf-8")
        sh(["docker", "build", "-t", tag, d], timeout=1800)
    return tag


def openclaw_config(model_id: str) -> str:
    """A minimal container config from the host's: the Workers AI provider (with its
    key - written into the container only), one model, exec without prompts."""
    host = json.loads((Path.home() / ".openclaw" / "openclaw.json").read_text(encoding="utf-8"))
    provider = host["models"]["providers"][PROVIDER]
    ref = f"{PROVIDER}/{model_id}"
    return json.dumps({
        "models": {"providers": {PROVIDER: provider}},
        "agents": {"defaults": {"model": {"primary": ref}, "models": {ref: {}}, "workspace": OC_WORKSPACE,
                                "modelPolicy": {"allow": [f"{PROVIDER}/*"]}}},
        "tools": {"profile": "coding", "exec": {"mode": "full"}},
    })


class Container:
    def __init__(self, image: str, memory_mb: int, workdir: str):
        self.name = f"xyb-tb-{uuid.uuid4().hex[:10]}"
        self.workdir = workdir
        sh(["docker", "run", "-d", "--name", self.name, "--memory", f"{memory_mb}m", "-w", workdir, image,
            "sleep", "infinity"])

    def exec(self, cmd: str, timeout: int, workdir: str | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
        args = ["docker", "exec", "-w", workdir or self.workdir]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        return sh(args + [self.name, "bash", "-lc", cmd], timeout=timeout, check=False)

    def put(self, text: str, path: str) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tmp", encoding="utf-8") as f:
            f.write(text)
            local = f.name
        try:
            self.exec(f"mkdir -p {Path(path).parent.as_posix()}", 60, workdir="/")
            sh(["docker", "cp", local, f"{self.name}:{path}"])
        finally:
            Path(local).unlink(missing_ok=True)

    def put_dir(self, local: Path, path: str) -> None:
        self.exec(f"mkdir -p {path}", 60, workdir="/")
        sh(["docker", "cp", f"{local}/.", f"{self.name}:{path}"])

    def commit(self) -> str:
        tag = f"xyb-tb-snap:{self.name}"
        sh(["docker", "commit", self.name, tag], timeout=900)
        return tag

    def remove(self) -> None:
        sh(["docker", "rm", "-f", self.name], check=False)


def grade(task: Task, c: Container) -> dict:
    """The task's own verifier: tests/ at /tests, reward from /logs/verifier/reward.txt."""
    c.put_dir(task.path / "tests", "/tests")
    c.exec("mkdir -p /logs/verifier && rm -f /logs/verifier/reward.txt", 60, workdir="/")
    t0 = time.time()
    out = c.exec("bash /tests/test.sh", task.verifier_timeout + 120)
    reward = c.exec("cat /logs/verifier/reward.txt 2>/dev/null", 60, workdir="/").stdout.strip()
    try:
        value = float(reward)
    except ValueError:
        value = None
    return {"reward": value, "seconds": round(time.time() - t0, 1), "tail": (out.stdout + out.stderr)[-400:]}


def validate(task: Task) -> dict:
    """Reference solution passes; the untouched environment fails."""
    image = task.base_image()
    wd = task.workdir(image)
    result = {}
    for label, solve in (("solution", True), ("untouched", False)):
        c = Container(image, task.memory_mb, wd)
        try:
            if solve:
                c.put_dir(task.path / "solution", "/solution")
                c.exec("bash /solution/solve.sh", task.agent_timeout)
            result[label] = grade(task, c)["reward"]
        finally:
            c.remove()
    result["valid"] = result["solution"] == 1.0 and result["untouched"] == 0.0
    return result


def agent_turn(c: Container, model_id: str, session: str, message: str, timeout: int) -> dict:
    c.put(message, "/tmp/xyb-message.txt")
    t0 = time.time()
    p = c.exec(f"openclaw agent --local --session-key agent:main:{session} --model {PROVIDER}/{model_id} "
               f"--timeout {timeout} --message-file /tmp/xyb-message.txt --json", timeout + 180, workdir=OC_WORKSPACE)
    return {"exit": p.returncode, "seconds": round(time.time() - t0, 1), "stdout": p.stdout,
            "stderr_tail": p.stderr[-800:]}


def transcript(c: Container) -> list[dict]:
    """The container's session messages, from a copy of its OpenClaw store."""
    from scenarios.export_calls import _events
    with tempfile.TemporaryDirectory() as d:
        sh(["docker", "cp", f"{c.name}:{OC_STORE}", d])
        conn = sqlite3.connect(Path(d) / "agent" / "openclaw-agent.sqlite")
        try:
            return [e["message"] for e in _events(conn, f"agent:main:{SESSION}") if e.get("type") == "message"]
        finally:
            conn.close()


def _find(o, key: str):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            if (r := _find(v, key)) is not None:
                return r
    return None


def run_summary(stdout: str) -> dict:
    """stopReason, tool summary and reply from `openclaw agent --json` output."""
    try:
        data = json.loads(stdout)
    except ValueError:
        return {}
    payloads = _find(data, "payloads") or []
    reply = "\n\n".join(p["text"] for p in payloads if isinstance(p, dict) and isinstance(p.get("text"), str))
    return {"stop_reason": (_find(data, "completion") or {}).get("stopReason"), "tools": _find(data, "toolSummary"),
            "reply": reply[-2000:] or None}


def start_agent(task: Task, model_id: str, turn_timeout: int) -> tuple[Container, dict]:
    """A configured container of the agent image, after the task's first turn."""
    image = agent_image(task)
    wd = task.workdir(image)
    c = Container(image, task.memory_mb, wd)
    try:
        c.put(openclaw_config(model_id), "/root/.openclaw/openclaw.json")
        c.exec(f"mkdir -p {OC_WORKSPACE}", 60, workdir="/")
        message = (f"This is an automated task. The files are in {wd} - work there. Don't ask questions; make "
                   f"reasonable assumptions and finish.\n\nTask: {task.instruction}")
        return c, agent_turn(c, model_id, SESSION, message, min(task.agent_timeout, turn_timeout))
    except BaseException:
        c.remove()
        raise


def drop_images(task: Task) -> None:
    """The task's own images; the shared runtime and base OS images stay."""
    for tag in (f"xyb-tb-agent:{task.name}", task.image or f"xyb-tb-env:{task.name}"):
        sh(["docker", "rmi", "-f", tag], check=False)


def survey_task(task: Task, model_id: str, turn_timeout: int) -> dict:
    """Grader check, then one first turn: does the model solve it on its own?"""
    rec = {"task": task.rel, "model": model_id, "started": datetime.now(timezone.utc).isoformat()}
    try:
        rec["grader"] = validate(task)
        if not rec["grader"]["valid"]:
            return rec
        c, turn = start_agent(task, model_id, turn_timeout)
        try:
            from scenarios.cf_features import features, tokens
            msgs = transcript(c)
            rec["first"] = {"exit": turn["exit"], "seconds": turn["seconds"], **run_summary(turn["stdout"]),
                            "tokens": tokens(msgs), "features": features(msgs)}
            if turn["exit"] != 0:
                rec["first"]["stderr_tail"] = turn["stderr_tail"]
            rec["reward"] = grade(task, c)["reward"]
        finally:
            c.remove()
    except Exception:  # noqa: BLE001 - one broken task must not stop the survey
        rec["error"] = traceback.format_exc()[-1500:]
    finally:
        drop_images(task)
    return rec


def survey(model_id: str, concurrency: int, turn_timeout: int, only: list[str] | None = None) -> None:
    """Every candidate task once. Resumable: tasks already surveyed without an error are skipped."""
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    log = SURVEY_DIR / "survey.jsonl"
    done = set()
    if log.exists():
        done = {(r["task"], r["model"]) for r in map(json.loads, log.read_text(encoding="utf-8").splitlines())
                if "error" not in r}
    todo = [t for t in (only or candidates()) if (t, model_id) not in done]
    print(f"{len(todo)} tasks to survey ({len(done)} done)", flush=True)
    ensure_runtime()
    with ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(survey_task, Task(t), model_id, turn_timeout) for t in todo]
        for i, f in enumerate(as_completed(futures), 1):
            rec = f.result()
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            first = rec.get("first") or {}
            status = "error" if "error" in rec else "bad-grader" if not rec["grader"]["valid"] else rec.get("reward")
            print(f"[{i}/{len(todo)}] {rec['task']:<44} {status!s:<10} {first.get('seconds', '-')!s:>7}s "
                  f"{first.get('tokens', '-')!s:>8} tok  {first.get('stop_reason') or ''}", flush=True)


def prototype(task: Task, model_id: str) -> dict:
    """One task end to end: first turn, snapshot, a same-session verify branch from
    the snapshot, and grades for the snapshot and the branch."""
    from scenarios.run_suite import FOLLOW_UPS
    c, first = start_agent(task, model_id, 900)
    wd = c.workdir
    out = {"task": task.rel, "model": model_id, "workdir": wd, "first": {**first, "stdout": first["stdout"][-1500:]}}
    try:
        snap = c.commit()
        out["snapshot"] = snap
    finally:
        c.remove()
    for label, follow in (("before", None), ("verify-same", FOLLOW_UPS["verify"])):
        b = Container(snap, task.memory_mb, wd)
        try:
            if follow:
                turn = agent_turn(b, model_id, SESSION, follow, min(task.agent_timeout, 900))
                out[f"{label}_turn"] = {**turn, "stdout": turn["stdout"][-1500:]}
            out[label] = grade(task, b)
        finally:
            b.remove()
    sh(["docker", "rmi", snap], check=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["validate", "prototype", "survey", "list"])
    parser.add_argument("--task", help="path under ~/tbench, e.g. archive/fix-git (survey: a comma list)")
    parser.add_argument("--model", default="glm53f")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--turn-timeout", type=int, default=900)
    args = parser.parse_args()
    from scenarios.run_suite import MODELS
    model_id = MODELS[args.model][0]
    if args.command == "list":
        print("\n".join(candidates()))
    elif args.command == "survey":
        survey(model_id, args.concurrency, args.turn_timeout, args.task.split(",") if args.task else None)
    elif not args.task:
        parser.error("--task is required")
    elif args.command == "validate":
        print(json.dumps(validate(Task(args.task)), indent=1))
    else:
        print(json.dumps(prototype(Task(args.task), model_id), indent=1))


if __name__ == "__main__":
    main()
