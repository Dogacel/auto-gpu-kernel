"""Kbench-owned execution for an agent-generated repository harness.

The agent supplies two Python adapters in ``harness/``. Kbench owns everything around
them: quick/full selection, validation-before-measurement, output parsing, A/B ordering,
history, timeouts, and provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from kbench.config import TaskConfig


VALIDATE_SCRIPT = "validate.py"
BENCHMARK_SCRIPT = "benchmark.py"
PREPARED_FILE = "prepared.json"
MODES = ("quick", "full")
TIMEOUT_S = {"quick": 1800, "full": 3600}


@dataclass
class TaskResult:
    mode: str
    exit_code: int
    seconds_wall: float
    validation_status: str
    error: str = ""
    value: float | None = None
    unit: str = ""
    lower_is_better: bool = True
    samples: list[float] = field(default_factory=list)
    out_dir: str = ""
    log_path: str = ""
    workdir_rev: str = ""
    harness_rev: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.exit_code == 0
            and self.validation_status == "pass"
            and not self.error
            and self.value is not None
        )


def workdir_rev(work: Path) -> str:
    """HEAD sha + dirty state hash, so a result identifies the candidate it ran."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=work,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=work,
            capture_output=True,
            check=True,
        ).stdout
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
            cwd=work,
            capture_output=True,
            check=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=work,
            capture_output=True,
            check=True,
        ).stdout.split(b"\0")
        if diff or status:
            digest = hashlib.sha256(diff)
            digest.update(status)
            for raw_path in sorted(path for path in untracked if path):
                path = work / os.fsdecode(raw_path)
                digest.update(len(raw_path).to_bytes(4, "big"))
                digest.update(raw_path)
                if path.is_symlink():
                    digest.update(os.fsencode(os.readlink(path)))
                elif path.is_file():
                    digest.update(path.read_bytes())
            return f"{head}+{digest.hexdigest()[:8]}"
        return head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def harness_dir(cfg: TaskConfig) -> Path:
    return cfg.root / "harness"


def prepared_path(cfg: TaskConfig) -> Path:
    return harness_dir(cfg) / PREPARED_FILE


def harness_is_prepared(cfg: TaskConfig) -> bool:
    return prepared_path(cfg).is_file()


def ensure_harness(cfg: TaskConfig) -> None:
    missing = [
        str(harness_dir(cfg) / name)
        for name in (VALIDATE_SCRIPT, BENCHMARK_SCRIPT)
        if not (harness_dir(cfg) / name).is_file()
    ]
    if missing:
        raise SystemExit(
            "generated task harness is incomplete; missing:\n  " + "\n  ".join(missing)
        )


def harness_rev(cfg: TaskConfig) -> str:
    """Hash functional harness inputs; edits are allowed but split result history."""
    ensure_harness(cfg)
    root = harness_dir(cfg)
    digest = hashlib.sha256()
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == PREPARED_FILE:
            continue
        rel = path.relative_to(root)
        if (
            "__pycache__" in rel.parts
            or ".pytest_cache" in rel.parts
            or path.suffix in {".pyc", ".pyo"}
            or path.name == ".DS_Store"
        ):
            continue
        files.append(path)
    for path in sorted(files):
        rel = str(path.relative_to(root)).encode()
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def mark_prepared(cfg: TaskConfig, results: list[TaskResult]) -> Path:
    """Record the pristine quick/full baselines after the builder turn."""
    path = prepared_path(cfg)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "harness_rev": harness_rev(cfg),
                "baseline": {
                    result.mode: {
                        "value": result.value,
                        "unit": result.unit,
                        "lower_is_better": result.lower_is_better,
                        "workdir_rev": result.workdir_rev,
                    }
                    for result in results
                },
            },
            indent=2,
        )
        + "\n"
    )
    return path


def _env(cfg: TaskConfig, extra: dict[str, str] | None) -> dict[str, str]:
    env = dict(os.environ)
    if cfg.path_prepend:
        env["PATH"] = f"{cfg.path_prepend}:{env.get('PATH', '')}"
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(cfg.gpus)))
    env["KBENCH_ROOT"] = str(cfg.root)
    env["KBENCH_HARNESS"] = str(harness_dir(cfg))
    env.update(cfg.env)
    env.update(extra or {})
    return env


def _run_script(
    cfg: TaskConfig,
    script: Path,
    work: Path,
    mode: str,
    output: Path,
    log_path: Path,
    *,
    append: bool,
    extra_env: dict[str, str] | None,
) -> tuple[int, float]:
    cmd = [
        sys.executable,
        str(script),
        "--repo",
        str(work),
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    print(f"[kbench] {script.stem}: {shlex.join(cmd)}")
    started = time.monotonic()
    timed_out = False
    with log_path.open("a" if append else "w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=work,
            env=_env(cfg, extra_env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        def kill() -> None:
            nonlocal timed_out
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except OSError:
                proc.kill()

        import threading

        watchdog = threading.Timer(TIMEOUT_S[mode], kill)
        watchdog.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            exit_code = proc.wait(timeout=60)
        except KeyboardInterrupt:
            kill()
            raise
        finally:
            watchdog.cancel()
            if proc.stdout is not None:
                proc.stdout.close()
        if timed_out:
            print(f"[kbench] KILLED: {script.name} exceeded {TIMEOUT_S[mode]}s")
            exit_code = exit_code or 124
    return exit_code, time.monotonic() - started


def _read_object(path: Path, label: str) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, f"{label} did not write {path.name}"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid {label} JSON: {exc}"
    if not isinstance(value, dict):
        return None, f"{label} output must be a JSON object"
    return value, None


def _metric(data: dict) -> tuple[float, str, bool, list[float]]:
    value = data.get("value")
    unit = data.get("unit")
    lower = data.get("lower_is_better")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("benchmark output needs numeric `value`")
    if not math.isfinite(float(value)):
        raise ValueError("benchmark `value` must be finite")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("benchmark output needs non-empty `unit`")
    if not isinstance(lower, bool):
        raise ValueError("benchmark output needs boolean `lower_is_better`")
    raw_samples = data.get("samples", [])
    if not isinstance(raw_samples, list):
        raise ValueError("benchmark `samples` must be a list when present")
    samples = []
    for sample in raw_samples:
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise ValueError("every benchmark sample must be numeric")
        if not math.isfinite(float(sample)):
            raise ValueError("every benchmark sample must be finite")
        samples.append(float(sample))
    return float(value), unit.strip(), lower, samples


def _integrity_error(
    cfg: TaskConfig,
    work: Path,
    candidate_rev: str,
    harness_hash: str,
) -> str:
    """Adapters may inspect and execute inputs, but must not rewrite them."""
    try:
        current_harness = harness_rev(cfg)
    except SystemExit:
        current_harness = "missing"
    if current_harness != harness_hash:
        return "task harness changed while kbench was running it"
    if workdir_rev(work) != candidate_rev:
        return "target repository changed while kbench was running it"
    return ""


def run_mode(
    cfg: TaskConfig,
    mode: str,
    *,
    workdir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    label: str = "",
) -> TaskResult:
    if mode not in MODES:
        raise SystemExit(f"unknown task mode {mode!r}; choose quick or full")
    ensure_harness(cfg)
    work = workdir or cfg.work
    if not work.exists():
        raise SystemExit(f"workdir not found: {work}")

    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
    out = cfg.root / ".kbench" / "out" / f"{stamp}-{mode}"
    if label:
        out = out.with_name(out.name + f"-{label}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    rev = workdir_rev(work)
    hrev = harness_rev(cfg)
    print(f"[kbench] mode={mode} cwd={work} candidate={rev} harness={hrev}")

    validation_path = out / "validation.json"
    started = time.monotonic()
    validate_exit, _ = _run_script(
        cfg,
        harness_dir(cfg) / VALIDATE_SCRIPT,
        work,
        mode,
        validation_path,
        log_path,
        append=False,
        extra_env=extra_env,
    )
    validation, validation_error = _read_object(validation_path, "validation")
    if validate_exit != 0:
        validation_status = f"FAIL: validate.py exited {validate_exit}"
    elif validation_error:
        validation_status = f"FAIL: {validation_error}"
    elif validation.get("passed") is not True:
        detail = validation.get("details", "reported passed=false")
        validation_status = f"FAIL: {detail}"
    else:
        validation_status = "pass"

    result = TaskResult(
        mode=mode,
        exit_code=validate_exit if validate_exit else (0 if validation_status == "pass" else 1),
        seconds_wall=time.monotonic() - started,
        validation_status=validation_status,
        out_dir=str(out),
        log_path=str(log_path),
        workdir_rev=rev,
        harness_rev=hrev,
    )
    if integrity_error := _integrity_error(cfg, work, rev, hrev):
        result.exit_code = 1
        result.error = integrity_error
        return result
    if validation_status != "pass":
        return result

    benchmark_path = out / "benchmark.json"
    benchmark_exit, _ = _run_script(
        cfg,
        harness_dir(cfg) / BENCHMARK_SCRIPT,
        work,
        mode,
        benchmark_path,
        log_path,
        append=True,
        extra_env=extra_env,
    )
    result.seconds_wall = time.monotonic() - started
    result.exit_code = benchmark_exit
    if integrity_error := _integrity_error(cfg, work, rev, hrev):
        result.exit_code = 1
        result.error = integrity_error
        return result
    benchmark, benchmark_error = _read_object(benchmark_path, "benchmark")
    if benchmark_exit != 0:
        result.error = f"benchmark.py exited {benchmark_exit}"
        return result
    if benchmark_error:
        result.exit_code = 1
        result.error = benchmark_error
        return result
    try:
        result.value, result.unit, result.lower_is_better, result.samples = _metric(benchmark)
    except ValueError as exc:
        result.exit_code = 1
        result.error = str(exc)
    return result


def print_result(cfg: TaskConfig, result: TaskResult) -> None:
    print(
        f"\n{cfg.name}  [{result.mode}]  local / {cfg.gpu} x{cfg.gpus}"
        f"  candidate={result.workdir_rev}  harness={result.harness_rev}"
    )
    status = "PASS" if result.passed else "FAIL"
    print(
        f"  {status}  exit={result.exit_code}  validation={result.validation_status}"
        f"  wall={result.seconds_wall:.1f}s"
    )
    if result.error:
        print(f"  error: {result.error}")
    if result.samples:
        samples = result.samples
        print(
            f"  samples n={len(samples)}: min={min(samples):.6g}"
            f" mean={statistics.fmean(samples):.6g}"
            f" median={statistics.median(samples):.6g} max={max(samples):.6g}"
        )
    if result.value is not None:
        direction = "lower is better" if result.lower_is_better else "higher is better"
        print(f"  metric: {result.value:.6g} {result.unit}  ({direction})")
    print(f"  artifacts: {result.out_dir}")


def record(cfg: TaskConfig, result: TaskResult) -> None:
    """Append to the same .kopt/bench.jsonl timeline kernel mode uses."""
    try:
        entry = {
            "t": time.time(),
            "mode": result.mode,
            "kernel": result.workdir_rev,
            "harness": result.harness_rev,
            "definition": cfg.name,
            "backend": "local",
            "gpu": f"{cfg.gpu} x{cfg.gpus}",
            "num_workloads": 1,
            "num_passed": 1 if result.passed else 0,
            "passed": result.passed,
            "validation": result.validation_status,
            "error": result.error,
            "metric_value": result.value,
            "metric": result.unit,
            "lower_is_better": result.lower_is_better,
            "samples": result.samples,
            "sample_mean": statistics.fmean(result.samples) if result.samples else None,
            "sample_median": statistics.median(result.samples) if result.samples else None,
            "sample_min": min(result.samples) if result.samples else None,
            "sample_max": max(result.samples) if result.samples else None,
        }
        path = cfg.root / ".kopt" / "bench.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def run_ab(
    cfg: TaskConfig,
    a_ref: str,
    mode: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> tuple[TaskResult, TaskResult]:
    """Run the same current harness against a git ref and the current candidate."""
    work = cfg.work
    tmp = cfg.root / ".kbench" / "ab_worktree"
    if tmp.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(tmp)],
            cwd=work,
            capture_output=True,
        )
        shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(
        ["git", "worktree", "add", "--force", "--detach", str(tmp), a_ref],
        cwd=work,
        check=True,
    )
    try:
        print(f"\n=== A: {a_ref} (worktree) ===")
        a = run_mode(cfg, mode, workdir=tmp, extra_env=extra_env, label="a")
        print("\n=== B: current tree ===")
        b = run_mode(cfg, mode, extra_env=extra_env, label="b")
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(tmp)],
            cwd=work,
            capture_output=True,
        )
        shutil.rmtree(tmp, ignore_errors=True)
    return a, b


def print_ab(cfg: TaskConfig, a: TaskResult, b: TaskResult, a_label: str) -> None:
    print(
        f"\nA = {a_label}\nB = current tree"
        f"   [local / {cfg.gpu} x{cfg.gpus}, harness {b.harness_rev}]\n"
    )
    for name, result in (("A", a), ("B", b)):
        value = (
            f"{result.value:.6g} {result.unit}" if result.value is not None else "-"
        )
        print(
            f"  {name}: {'PASS' if result.passed else 'FAIL'}"
            f"  metric={value}  validation={result.validation_status}"
            + (f"  error={result.error}" if result.error else "")
        )
    if not a.passed or not b.passed:
        return
    if a.harness_rev != b.harness_rev:
        print("\n  REFUSED: A and B used different harness revisions")
        return
    if a.unit != b.unit or a.lower_is_better != b.lower_is_better:
        print("\n  REFUSED: A and B reported incompatible metrics")
        return
    if a.value is None or b.value is None or a.value == 0:
        return
    delta = b.value - a.value
    pct = 100.0 * delta / abs(a.value)
    b_better = delta < 0 if b.lower_is_better else delta > 0
    verdict = "B better" if b_better else "A better" if delta else "tie"
    print(f"\n  B-A = {delta:+.6g} {b.unit} ({pct:+.2f}%) -> {verdict}")
    if abs(pct) < 2.0:
        print("  NOTE: delta under 2% — treat as noise unless it reproduces")
