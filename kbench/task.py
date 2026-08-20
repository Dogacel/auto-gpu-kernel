"""Generic task benchmarking — run a repo-native bench command, verify, measure.

Task mode replaces the flashinfer-bench harness with the target repo's own
verification entry points. The contract that keeps results trustworthy:

- The verifier is **pinned**: harness-owned copies (``[[task.pinned]]``) are written
  fresh into the workdir before every run, so edits to the working tree cannot
  change what "correct" means.
- Correctness is **exact-artifact equality**: the produced artifact JSON (e.g. a
  per-step digest ledger) must equal a golden captured from the unmodified repo,
  and the command itself must exit 0.
- The metric is parsed from the pinned verifier's own output, never from
  agent-written files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from kbench.config import BenchMode, TaskConfig


@dataclass
class TaskResult:
    mode: str
    exit_code: int
    seconds_wall: float
    value: float | None = None
    """Headline metric in seconds (lower is better unless config says otherwise)."""
    steps_ms: list[float] = field(default_factory=list)
    golden_status: str = "n/a"  # "match" | "MISMATCH: ..." | "captured" | "n/a"
    out_dir: str = ""
    log_path: str = ""
    workdir_rev: str = ""

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.golden_status.startswith("MISMATCH")


def workdir_rev(work: Path) -> str:
    """HEAD sha + dirty-diff hash, so a measurement pins the exact tree it ran on."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=work, capture_output=True, text=True, check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=work, capture_output=True, text=True, check=True,
        ).stdout
        if diff:
            import hashlib

            return f"{head}+{hashlib.sha256(diff.encode()).hexdigest()[:8]}"
        return head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _env(cfg: TaskConfig, extra: dict[str, str] | None) -> dict[str, str]:
    env = dict(os.environ)
    if cfg.path_prepend:
        env["PATH"] = f"{cfg.path_prepend}:{env.get('PATH', '')}"
    # All GPUs visible by default; scripts that setdefault CUDA_VISIBLE_DEVICES
    # would otherwise pin themselves to one device.
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(cfg.gpus)))
    env.update(cfg.env)
    env.update(extra or {})
    return env


def _place_pinned(cfg: TaskConfig, work: Path) -> None:
    _exclude_locally(work, {("/" + p.dst.split("/")[0].rstrip("/") + ("/" if "/" in p.dst else ""))
                            for p in cfg.pinned})
    for pin in cfg.pinned:
        src = cfg.root / pin.src
        if not src.exists():
            raise SystemExit(f"pinned file missing: {src} — re-run `kopt init-task`")
        dst = work / pin.dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def _exclude_locally(work: Path, patterns: set[str]) -> None:
    """Keep pinned copies out of the target repo's history via .git/info/exclude
    (local-only, so the working tree diff stays clean of harness files)."""
    exclude = work / ".git" / "info" / "exclude"
    if not exclude.parent.is_dir():
        return
    have = exclude.read_text().splitlines() if exclude.exists() else []
    missing = [p for p in sorted(patterns) if p not in have]
    if missing:
        exclude.write_text("\n".join([*have, *missing]) + "\n")


def _dotted(obj, path: str):
    for part in path.split("."):
        obj = obj[part]
    return obj


def _compare_golden(golden_path: Path, artifact_path: Path) -> str:
    """Exact JSON equality, with a first-difference pointer on mismatch."""
    if not artifact_path.exists():
        return f"MISMATCH: artifact {artifact_path.name} was not produced"
    if not golden_path.exists():
        return f"MISMATCH: golden {golden_path} missing (capture with --capture-golden)"
    golden = json.loads(golden_path.read_text())
    got = json.loads(artifact_path.read_text())
    if golden == got:
        return "match"
    return f"MISMATCH: {_first_diff(golden, got, '$')}"


def _first_diff(a, b, path: str) -> str:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in a:
            if k not in b:
                return f"{path}.{k}: missing"
            if a[k] != b[k]:
                return _first_diff(a[k], b[k], f"{path}.{k}")
        for k in b:
            if k not in a:
                return f"{path}.{k}: unexpected"
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _first_diff(x, y, f"{path}[{i}]")
    else:
        return f"{path}: {a!r} != {b!r}"
    return f"{path}: differs"


def run_mode(
    cfg: TaskConfig,
    mode: BenchMode,
    *,
    workdir: Path | None = None,
    capture_golden: bool = False,
    extra_env: dict[str, str] | None = None,
    label: str = "",
) -> TaskResult:
    work = workdir or cfg.work
    if not work.exists():
        raise SystemExit(f"workdir not found: {work}")
    _place_pinned(cfg, work)

    out = cfg.root / ".kbench" / "out" / f"{time.strftime('%Y%m%d-%H%M%S')}-{mode.name}"
    if label:
        out = out.with_name(out.name + f"-{label}")
    out.mkdir(parents=True, exist_ok=True)

    cmd = mode.cmd.format(out=str(out))
    rev = workdir_rev(work)
    print(f"[kbench] {mode.name}: {cmd}")
    print(f"[kbench] cwd={work} rev={rev} gpus={cfg.gpus}")

    log_path = out / "bench.log"
    lines: list[str] = []
    t0 = time.monotonic()
    timed_out = False
    with log_path.open("w") as log:
        # start_new_session so a timeout kill reaps the whole process tree (uv -> python).
        proc = subprocess.Popen(
            cmd, shell=True, cwd=work, env=_env(cfg, extra_env),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )

        def _kill():
            nonlocal timed_out
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (OSError, ProcessLookupError):
                proc.kill()

        import threading

        watchdog = threading.Timer(mode.timeout_s, _kill)
        watchdog.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                lines.append(line)
            exit_code = proc.wait(timeout=60)
        except KeyboardInterrupt:
            _kill()
            raise
        finally:
            watchdog.cancel()
        if timed_out:
            print(f"[kbench] KILLED: run exceeded timeout_s={mode.timeout_s}")
            exit_code = exit_code or 124
    seconds_wall = time.monotonic() - t0
    stdout = "".join(lines)

    result = TaskResult(
        mode=mode.name, exit_code=exit_code, seconds_wall=seconds_wall,
        out_dir=str(out), log_path=str(log_path), workdir_rev=rev,
    )

    # --- metric -----------------------------------------------------------
    if mode.step_regex:
        result.steps_ms = [
            float(m.group("ms")) for m in re.finditer(mode.step_regex, stdout)
        ]
    if mode.metric_regex:
        matches = list(re.finditer(mode.metric_regex, stdout))
        if matches:
            result.value = float(matches[-1].group("value"))
    elif mode.metric_json and len(mode.metric_json) == 2:
        art = out / mode.metric_json[0]
        if art.exists():
            try:
                result.value = float(_dotted(json.loads(art.read_text()), mode.metric_json[1])) / 1000.0
            except (KeyError, TypeError, ValueError):
                pass
    elif result.steps_ms:
        result.value = sum(result.steps_ms) / 1000.0
    if result.value is None and result.steps_ms:
        result.value = sum(result.steps_ms) / 1000.0

    # --- correctness --------------------------------------------------------
    if mode.golden and mode.artifact:
        artifact = out / mode.artifact
        golden = cfg.root / mode.golden
        if capture_golden:
            if exit_code == 0 and artifact.exists():
                golden.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(artifact, golden)
                result.golden_status = "captured"
            else:
                result.golden_status = "MISMATCH: cannot capture golden from a failed run"
        else:
            result.golden_status = _compare_golden(golden, artifact)

    return result


def print_result(cfg: TaskConfig, r: TaskResult) -> None:
    print(f"\n{cfg.name}  [{r.mode}]  local / {cfg.gpu} x{cfg.gpus}  rev={r.workdir_rev}")
    status = "PASS" if r.passed else "FAIL"
    print(f"  {status}  exit={r.exit_code}  golden={r.golden_status}  wall={r.seconds_wall:.1f}s")
    if r.steps_ms:
        s = r.steps_ms
        print(
            f"  forwards n={len(s)}: sum={sum(s) / 1000:.3f}s"
            f" | min {min(s):.0f} / mean {statistics.fmean(s):.0f}"
            f" / median {statistics.median(s):.0f} / max {max(s):.0f} ms"
        )
    if r.value is not None:
        print(f"  metric: {r.value:.3f} s  ({cfg.metric})")
    print(f"  artifacts: {r.out_dir}")


def record(cfg: TaskConfig, r: TaskResult) -> None:
    """Append to the same .kopt/bench.jsonl timeline kernel mode uses."""
    try:
        entry = {
            "t": time.time(),
            "mode": r.mode,
            "kernel": r.workdir_rev,
            "definition": cfg.name,
            "backend": "local",
            "gpu": f"{cfg.gpu} x{cfg.gpus}",
            "num_workloads": max(len(r.steps_ms), 1),
            "num_passed": (max(len(r.steps_ms), 1)) if r.passed else 0,
            "passed": r.passed,
            "golden": r.golden_status,
            "value_s": r.value,
            "mean_latency_ms": statistics.fmean(r.steps_ms) if r.steps_ms else None,
            "median_latency_ms": statistics.median(r.steps_ms) if r.steps_ms else None,
            "min_latency_ms": min(r.steps_ms) if r.steps_ms else None,
            "max_latency_ms": max(r.steps_ms) if r.steps_ms else None,
        }
        path = cfg.root / ".kopt" / "bench.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # observability must never take down a run


def run_ab(cfg: TaskConfig, a_ref: str, mode: BenchMode) -> tuple[TaskResult, TaskResult]:
    """A/B: benchmark a past git ref in a temporary worktree, then the current tree.

    Same machine, same GPUs, back-to-back — the closest task mode gets to a paired
    comparison. The A worktree may pay a one-time extension/JIT rebuild; per-step
    numbers are unaffected (build happens during load).
    """
    work = cfg.work
    tmp = cfg.root / ".kbench" / "ab_worktree"
    if tmp.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(tmp)],
                       cwd=work, capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(
        ["git", "worktree", "add", "--force", "--detach", str(tmp), a_ref],
        cwd=work, check=True,
    )
    try:
        print(f"\n=== A: {a_ref} (worktree) ===")
        a = run_mode(cfg, mode, workdir=tmp, label="a")
        print(f"\n=== B: current tree ===")
        b = run_mode(cfg, mode, label="b")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(tmp)],
                       cwd=work, capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return a, b


def print_ab(cfg: TaskConfig, a: TaskResult, b: TaskResult, a_label: str) -> None:
    print(f"\nA = {a_label}\nB = current tree   [local / {cfg.gpu} x{cfg.gpus}]\n")
    for name, r in (("A", a), ("B", b)):
        v = f"{r.value:.3f}s" if r.value is not None else "-"
        print(f"  {name}: {'PASS' if r.passed else 'FAIL'}  metric={v}  golden={r.golden_status}")
    if a.value and b.value:
        d = b.value - a.value
        pct = 100.0 * d / a.value
        verdict = "B faster" if d < 0 else "A faster" if d > 0 else "tie"
        print(f"\n  B-A = {d:+.3f}s ({pct:+.2f}%) -> {verdict}")
        if abs(pct) < 2.0:
            print("  NOTE: delta under 2% — treat as noise unless it reproduces")
