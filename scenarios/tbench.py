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
import posixpath
import shlex
import sqlite3
import subprocess
import tempfile
import threading
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
        self.separate = (self.cfg.get("verifier") or {}).get("environment_mode") == "separate"
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
            if list((t / "environment").glob("docker-compose*")) or any(
                    isinstance(a, dict) and a.get("service") for a in cfg.get("artifacts") or []):
                continue  # multi-container tasks need compose; this harness runs one container
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


def openclaw_config(model_id: str, extra_models: tuple[str, ...] = (), max_tokens: int | None = None) -> str:
    """A minimal container config from the host's: the Workers AI provider (with its
    key - written into the container only), the run's model (plus any a branch
    switches to), exec without prompts. max_tokens overrides the output cap for
    the run's models (the host sets 8192; every length-stopped message in the
    2026-09 experiments stopped at exactly that, and OpenClaw then substitutes
    its fallback reply - the commonest death)."""
    host = json.loads((Path.home() / ".openclaw" / "openclaw.json").read_text(encoding="utf-8"))
    provider = json.loads(json.dumps(host["models"]["providers"][PROVIDER]))  # a copy: the host config isn't ours to change
    if max_tokens:
        for m in provider.get("models", []):
            if m.get("id") in (model_id, *extra_models):
                m["maxTokens"] = max_tokens
    ref = f"{PROVIDER}/{model_id}"
    return json.dumps({
        "models": {"providers": {PROVIDER: provider}},
        "agents": {"defaults": {"model": {"primary": ref}, "workspace": OC_WORKSPACE,
                                "models": {f"{PROVIDER}/{m}": {} for m in (model_id, *extra_models)},
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


def _run_tests(task: Task, c: Container) -> dict:
    c.exec("mkdir -p /logs/verifier && rm -f /logs/verifier/reward.txt", 60, workdir="/")
    t0 = time.time()
    out = c.exec("bash /tests/test.sh", task.verifier_timeout + 120)
    reward = c.exec("cat /logs/verifier/reward.txt 2>/dev/null", 60, workdir="/").stdout.strip()
    try:
        value = float(reward)
    except ValueError:
        value = None
    return {"reward": value, "seconds": round(time.time() - t0, 1), "tail": (out.stdout + out.stderr)[-400:]}


def verifier_image(task: Task) -> str:
    """A separate-mode task's verifier environment, built from tests/ (it holds /tests itself)."""
    tag = f"xyb-tb-verifier:{task.name}"
    if sh(["docker", "image", "inspect", tag], check=False).returncode != 0:
        sh(["docker", "build", "-t", tag, str(task.path / "tests")], timeout=3600)
    return tag


def copy_artifact(src: Container, dst: Container, artifact) -> None:
    """One declared artifact (a path, or {source, exclude}) from the agent's container
    to the verifier's, streamed as a tar. One the agent never made is skipped."""
    path = (artifact["source"] if isinstance(artifact, dict) else artifact).rstrip("/")
    parent = posixpath.dirname(path) or "/"
    dst.exec(f"mkdir -p {shlex.quote(parent)}", 60, workdir="/")
    out = subprocess.Popen(["docker", "cp", f"{src.name}:{path}", "-"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
    try:
        subprocess.run(["docker", "cp", "-", f"{dst.name}:{parent}"], stdin=out.stdout, capture_output=True,
                       timeout=900)
    finally:
        out.stdout.close()
        out.wait(timeout=60)
    for pattern in (artifact.get("exclude") or []) if isinstance(artifact, dict) else []:
        name = posixpath.basename(pattern.rstrip("/"))
        dst.exec(f"find {shlex.quote(path)} -name {shlex.quote(name)} -prune -exec rm -rf {{}} +", 120, workdir="/")


def grade(task: Task, c: Container) -> dict:
    """The task's own verifier, reward from /logs/verifier/reward.txt. Shared mode
    runs tests/ inside the agent's container; separate mode runs them in the
    verifier image, with the task's declared artifacts copied over from it."""
    if not task.separate:
        c.put_dir(task.path / "tests", "/tests")
        return _run_tests(task, c)
    image = verifier_image(task)
    v = Container(image, task.memory_mb, task.workdir(image))
    try:
        for artifact in task.cfg.get("artifacts") or []:
            copy_artifact(c, v, artifact)
        return _run_tests(task, v)
    finally:
        v.remove()


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
    """One `openclaw agent` turn. status is run_suite's: done (exit 0), failed (the
    run died), timeout (it didn't return in time)."""
    c.put(message, "/tmp/xyb-message.txt")
    t0 = time.time()
    try:
        p = c.exec(f"openclaw agent --local --session-key agent:main:{session} --model {PROVIDER}/{model_id} "
                   f"--timeout {timeout} --message-file /tmp/xyb-message.txt --json", timeout + 180,
                   workdir=OC_WORKSPACE)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "exit": None, "seconds": round(time.time() - t0, 1), "stdout": "",
                "stderr_tail": ""}
    return {"status": "done" if p.returncode == 0 else "failed", "exit": p.returncode,
            "seconds": round(time.time() - t0, 1), "stdout": p.stdout, "stderr_tail": p.stderr[-800:]}


def transcript(c: Container, session: str = SESSION) -> list[dict]:
    """A session's messages, from a copy of the container's OpenClaw store."""
    from scenarios.export_calls import _events
    with tempfile.TemporaryDirectory() as d:
        sh(["docker", "cp", f"{c.name}:{OC_STORE}", d])
        conn = sqlite3.connect(Path(d) / "agent" / "openclaw-agent.sqlite")
        try:
            return [e["message"] for e in _events(conn, f"agent:main:{session}") if e.get("type") == "message"]
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


def task_message(task: Task, wd: str) -> str:
    return (f"This is an automated task. The files are in {wd} - work there. Don't ask questions; make "
            f"reasonable assumptions and finish.\n\nTask: {task.instruction}")


def start_agent(task: Task, model_id: str, turn_timeout: int, extra_models: tuple[str, ...] = (),
                max_tokens: int | None = None) -> tuple[Container, dict]:
    """A configured container of the agent image, after the task's first turn."""
    image = agent_image(task)
    wd = task.workdir(image)
    c = Container(image, task.memory_mb, wd)
    try:
        c.put(openclaw_config(model_id, extra_models, max_tokens), "/root/.openclaw/openclaw.json")
        c.exec(f"mkdir -p {OC_WORKSPACE}", 60, workdir="/")
        return c, agent_turn(c, model_id, SESSION, task_message(task, wd), min(task.agent_timeout, turn_timeout))
    except BaseException:
        c.remove()
        raise


def drop_images(task: Task) -> None:
    """The task's own images; the shared runtime and base OS images stay."""
    for tag in (f"xyb-tb-agent:{task.name}", f"xyb-tb-verifier:{task.name}", task.image or f"xyb-tb-env:{task.name}"):
        sh(["docker", "rmi", "-f", tag], check=False)


def survey_task(task: Task, model_id: str, turn_timeout: int, max_tokens: int | None = None) -> dict:
    """Grader check, then one first turn: does the model solve it on its own?"""
    rec = {"task": task.rel, "model": model_id, "started": datetime.now(timezone.utc).isoformat()}
    if max_tokens:
        rec["max_tokens"] = max_tokens
    try:
        rec["grader"] = validate(task)
        if not rec["grader"]["valid"]:
            return rec
        c, turn = start_agent(task, model_id, turn_timeout, max_tokens=max_tokens)
        try:
            from scenarios.cf_features import features, tokens
            msgs = transcript(c)
            rec["first"] = {"status": turn["status"], "exit": turn["exit"], "seconds": turn["seconds"],
                            **run_summary(turn["stdout"]),
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


def survey(model_id: str, concurrency: int, turn_timeout: int, only: list[str] | None = None,
           max_tokens: int | None = None) -> None:
    """Every candidate task once. Resumable: tasks already surveyed without an error are
    skipped. A max_tokens override gets its own log, so it never stands in for the baseline."""
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    log = SURVEY_DIR / (f"survey-max{max_tokens}.jsonl" if max_tokens else "survey.jsonl")
    done = set()
    if log.exists():
        done = {(r["task"], r["model"]) for r in map(json.loads, log.read_text(encoding="utf-8").splitlines())
                if "error" not in r}
    todo = [t for t in (only or candidates()) if (t, model_id) not in done]
    print(f"{len(todo)} tasks to survey ({len(done)} done)" + (f", max_tokens={max_tokens}" if max_tokens else ""), flush=True)
    ensure_runtime()
    with ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(survey_task, Task(t), model_id, turn_timeout, max_tokens) for t in todo]
        for i, f in enumerate(as_completed(futures), 1):
            rec = f.result()
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            first = rec.get("first") or {}
            status = "error" if "error" in rec else "bad-grader" if not rec["grader"]["valid"] else rec.get("reward")
            print(f"[{i}/{len(todo)}] {rec['task']:<44} {status!s:<10} {first.get('seconds', '-')!s:>7}s "
                  f"{first.get('tokens', '-')!s:>8} tok  {first.get('stop_reason') or ''}", flush=True)


def grade_image(task: Task, image: str, wd: str) -> dict:
    """Grade a fresh container of an image (grading installs things; it gets its own)."""
    c = Container(image, task.memory_mb, wd)
    try:
        return grade(task, c)
    finally:
        c.remove()


def _branch(task: Task, snap: str, wd: str, model: str, branch: tuple, n_first: int, turn_timeout: int) -> dict:
    """One follow-up in its own container from the snapshot: grade, tokens, time.
    A same-session branch continues the first turn's session (a true fork); a
    fresh one starts a new session with the task restated. Never raises."""
    from scenarios.cf_features import tokens
    from scenarios.run_suite import BRANCH_PROMPTS, FOLLOW_UPS, MODELS
    name, kind, strong, same = branch
    model = strong or model
    t0 = time.time()
    try:
        b = Container(snap, task.memory_mb, wd)
    except Exception as exc:  # noqa: BLE001
        return {"task": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
    try:
        session = SESSION if same else f"{SESSION}-{name}"
        message = FOLLOW_UPS[kind] if same else task_message(task, wd) + "\n\n" + BRANCH_PROMPTS[kind]
        turn = agent_turn(b, MODELS[model][0], session, message, min(task.agent_timeout, turn_timeout))
        msgs = transcript(b, session)
        reward = grade(task, b)["reward"]
        graded = None if reward is None else reward == 1.0
        # As in run_suite: a check that dies leaves the first answer standing; a retry that dies is a failure.
        ok = graded if (turn["status"] == "done" or kind != "retry") else False
        return {"task": ok, "graded": graded, "reward": reward, "status": turn["status"], "model": model,
                "tokens": tokens(msgs[n_first:] if same else msgs), "seconds": round(time.time() - t0, 1),
                "stop_reason": run_summary(turn["stdout"]).get("stop_reason"), "stderr_tail": turn["stderr_tail"][-300:]}
    except Exception as exc:  # noqa: BLE001 - one broken branch must not lose the run's other outcomes
        return {"task": None, "error": f"{type(exc).__name__}: {exc}"[:300], "seconds": round(time.time() - t0, 1)}
    finally:
        b.remove()


def cf_run(task: Task, model: str, rep: int, batch: str, turn_timeout: int) -> dict:
    """run_suite --counterfactual for one Terminal-Bench task: the first turn, the
    snapshot graded as it stands, every follow-up branch from that snapshot. The
    record has run_suite's manifest shape, so learn.uplift reads it unchanged."""
    from scenarios.cf_features import features, tokens
    from scenarios.run_suite import DEAD_BRANCHES, FINISHED_BRANCHES, MODELS, STRONG_MODEL
    rec = {"batch": batch, "scenario": f"tb:{task.name}", "category": "tbench", "task_path": task.rel,
           "model": model, "rep": rep, "session_key": f"tb:{batch}:{task.name}:{model}:r{rep}",
           "started": datetime.now(timezone.utc).isoformat()}
    snap = None
    try:
        c, first = start_agent(task, MODELS[model][0], turn_timeout, extra_models=(MODELS[STRONG_MODEL][0],))
        try:
            msgs = transcript(c)
            rec.update(status=first["status"], exit_code=first["exit"], seconds=first["seconds"],
                       **{k: v for k, v in run_summary(first["stdout"]).items() if k != "tools"})
            cf = {"features": features(msgs), "tokens_first": tokens(msgs), "kind": None, "branches": {}}
            snap = c.commit()
        finally:
            c.remove()
        branches = [] if first["status"] == "timeout" else \
            FINISHED_BRANCHES if first["status"] == "done" else DEAD_BRANCHES
        todo = [br for br in branches if not (br[2] and br[2] == model)]  # "-strong" on the strong model = "-fresh"
        wd = c.workdir
        with ThreadPoolExecutor(max_workers=len(todo) + 1) as pool:
            before = pool.submit(grade_image, task, snap, wd)
            outs = dict(zip([br[0] for br in todo], pool.map(
                lambda br: _branch(task, snap, wd, model, br, len(msgs), turn_timeout), todo)))
            g = before.result()
        passed = None if g["reward"] is None else g["reward"] == 1.0
        # A run that died gave the user no answer, whatever is on disk.
        cf["before"] = {"task": passed if first["status"] == "done" else False, "graded": passed, "reward": g["reward"]}
        if not branches:
            cf["note"] = "first turn timed out: no follow-up"
        else:
            cf["kind"] = "finished" if first["status"] == "done" else "died"
            for name, _kind, strong, _same in branches:
                fresh = name.replace("-strong", "-fresh")
                cf["branches"][name] = outs.get(name) or {**outs[fresh], "alias": fresh}
        rec["counterfactual"] = cf
    except Exception:  # noqa: BLE001 - one broken run must not stop the batch
        rec["status"], rec["error"] = "crashed", traceback.format_exc()[-1500:]
    finally:
        if snap:
            sh(["docker", "rmi", "-f", snap], check=False)
    return rec


def cf_batch(batch: str, tasks: list[str], models: list[str], reps: int, concurrency: int, turn_timeout: int) -> None:
    """Counterfactual runs for tasks x models x reps. A worker takes a whole task,
    so its images are dropped once, after its last run. Resumable: runs already in
    runs.jsonl without a crash are skipped."""
    from scenarios.run_suite import BATCH_DIR
    batch_dir = BATCH_DIR / batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "runs.jsonl"
    done = set()
    if manifest.exists():
        done = {r["session_key"] for r in map(json.loads, manifest.read_text(encoding="utf-8").splitlines())
                if r.get("status") != "crashed"}
    jobs = {t: [(m, r) for r in range(1, reps + 1) for m in models
                if f"tb:{batch}:{Path(t).name}:{m}:r{r}" not in done] for t in tasks}
    jobs = {t: j for t, j in jobs.items() if j}
    total = sum(map(len, jobs.values()))
    print(f"{total} runs over {len(jobs)} tasks to go ({len(done)} already in the manifest)", flush=True)
    ensure_runtime()
    lock, counter = threading.Lock(), [0]

    def work(t: str) -> None:
        task = Task(t)
        try:
            for m, r in jobs[t]:
                rec = cf_run(task, m, r, batch, turn_timeout)
                with lock:
                    counter[0] += 1
                    with open(manifest, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    cf = rec.get("counterfactual") or {}
                    arms = " ".join(f"{k}={'1' if v.get('task') else '0' if v.get('task') is False else '?'}"
                                    for k, v in cf.get("branches", {}).items())
                    before = (cf.get("before") or {}).get("task")
                    print(f"[{counter[0]}/{total}] {task.name:<34} {m:<8} r{r} {rec.get('status'):<8} "
                          f"before={before} {arms}", flush=True)
        finally:
            drop_images(task)

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(work, jobs))


def surveyed(valid_only: bool = True) -> list[str]:
    """Tasks in the survey log whose grader checked out."""
    rows = [json.loads(line) for line in (SURVEY_DIR / "survey.jsonl").read_text(encoding="utf-8").splitlines()]
    return sorted({r["task"] for r in rows if not valid_only or (r.get("grader") or {}).get("valid")})


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
    parser.add_argument("command", choices=["validate", "prototype", "survey", "list", "cf"])
    parser.add_argument("--task", help="path under ~/tbench, e.g. archive/fix-git (survey, cf: a comma list; "
                                       "cf defaults to every task the survey found a working grader for)")
    parser.add_argument("--model", default="glm53f", help="cf: a comma list of short names")
    parser.add_argument("--batch", help="cf: batch name under scenarios/batches")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--turn-timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, help="survey: override the models' output cap (host default 8192)")
    args = parser.parse_args()
    from scenarios.run_suite import MODELS
    if args.command == "cf":
        if not args.batch:
            parser.error("--batch is required")
        models = args.model.split(",")
        unknown = [m for m in models if m not in MODELS]
        if unknown:
            parser.error(f"unknown models {unknown}; known: {', '.join(MODELS)}")
        tasks = args.task.split(",") if args.task else surveyed()
        cf_batch(args.batch, tasks, models, args.reps, args.concurrency, args.turn_timeout)
        return
    model_id = MODELS[args.model][0]
    if args.command == "list":
        print("\n".join(candidates()))
    elif args.command == "survey":
        survey(model_id, args.concurrency, args.turn_timeout, args.task.split(",") if args.task else None,
               args.max_tokens)
    elif not args.task:
        parser.error("--task is required")
    elif args.command == "validate":
        print(json.dumps(validate(Task(args.task)), indent=1))
    else:
        print(json.dumps(prototype(Task(args.task), model_id), indent=1))


if __name__ == "__main__":
    main()
