"""kopt — scaffold a kernel project, then run the optimization loop over it."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from kbench import adapters
from kbench import config as bench_config
from kbench import task as taskmod
from kbench.adapters.base import RunRequest
from kbench.config import TaskConfig
from kopt.init import init, init_task, languages
from kopt.loop import DEFAULT_PROMPT, Loop
from kopt.record import list_run_logs
from kopt.watch import serve


def cmd_init(args) -> int:
    project = init(
        project=Path(args.project).resolve(),
        definition_json=Path(args.definition),
        language=args.language,
        backend=args.backend,
        gpu=args.gpu,
        force=args.force,
    )
    print(f"scaffolded {project}")
    print(f"  language: {args.language}   backend: {args.backend}/{args.gpu}")
    print(f"\nnext:  kopt run {project} -n 20 --budget 20")
    return 0


def cmd_init_task(args) -> int:
    project = init_task(
        project=Path(args.project).resolve(),
        taskspec=Path(args.taskspec),
        force=args.force,
    )
    print(f"scaffolded task project {project}")
    print("  the target repo is isolated inside the project")
    print("\nnext:")
    print(f"  kopt run {project} -n 20 --model anthropic/claude-opus-5 --thinking low")
    return 0


def _git_state(work: Path) -> tuple[str, str, str]:
    """HEAD, readable status, and an exact fingerprint of the dirty state."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=work, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=work, check=True, capture_output=True, text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=work, check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=work, check=True, capture_output=True,
        ).stdout.split(b"\0")
        digest = hashlib.sha256(diff)
        digest.update(status.encode())
        for raw_path in sorted(path for path in untracked if path):
            path = work / os.fsdecode(raw_path)
            digest.update(len(raw_path).to_bytes(4, "big"))
            digest.update(raw_path)
            if path.is_symlink():
                digest.update(os.fsencode(os.readlink(path)))
            elif path.is_file():
                digest.update(path.read_bytes())
        return head, status, digest.hexdigest()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"target workdir is not a usable git repository: {work} ({exc})") from exc


def _commit_generated_harness(cfg: TaskConfig) -> None:
    ident = ["-c", "user.email=kopt@localhost", "-c", "user.name=kopt"]
    try:
        subprocess.run(
            ["git", "add", "config.toml", ".omp", "harness"],
            cwd=cfg.root, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", *ident, "commit", "-q", "-m", "build generated harness"],
            cwd=cfg.root, check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"could not commit the generated harness: {exc}") from exc


def _prepare_task(cfg: TaskConfig, args) -> float:
    """Use one isolated turn to generate the adapters kbench will orchestrate."""
    if taskmod.harness_is_prepared(cfg):
        taskmod.ensure_harness(cfg)
        return 0.0

    before = _git_state(cfg.work)
    framework = Path(__file__).resolve().parents[1]
    framework_before = _git_state(framework) if (framework / ".git").exists() else None
    brief_path = cfg.root / "config.toml"
    brief = brief_path.read_bytes()
    if before[1]:
        raise SystemExit(
            "the target repo must be clean before its harness is built:\n" + before[1]
        )

    print("\nNo generated harness yet; building it from the task brief.")
    builder = Loop(
        project=cfg.root,
        prompt="/skill:build-harness",
        max_iterations=1,
        timeout=args.timeout,
        max_time=args.max_time,
        model=args.model,
        thinking=args.thinking,
        fresh=True,
    )
    builder.run()

    after_builder = _git_state(cfg.work)
    if framework_before is not None and _git_state(framework) != framework_before:
        raise SystemExit("harness builder changed the auto-gpu-kernel source checkout")
    if not brief_path.is_file() or brief_path.read_bytes() != brief:
        raise SystemExit("harness builder changed the user's task brief; refusing to continue")
    if after_builder != before:
        raise SystemExit(
            "harness builder changed the target repo; refusing to establish a baseline.\n"
            f"before HEAD: {before[0]}\nafter HEAD:  {after_builder[0]}\n"
            f"current status:\n{after_builder[1] or '(clean)'}"
        )

    taskmod.ensure_harness(cfg)
    generated_rev = taskmod.harness_rev(cfg)
    print("\nRunning pristine quick and full baselines through kbench.")
    adapter = adapters.get(cfg)
    baselines = []
    for mode in taskmod.MODES:
        measurement = adapter.bench(RunRequest(mode=mode))
        adapter.print_result(measurement)
        adapter.record(measurement)
        if not measurement.passed:
            raise SystemExit(f"generated harness failed its pristine {mode!r} run")
        baselines.append(measurement)
    if any(result.harness_rev != generated_rev for result in baselines):
        raise SystemExit("generated harness modified itself while kbench was running it")
    contracts = {(result.unit, result.lower_is_better) for result in baselines}
    if len(contracts) != 1:
        raise SystemExit("quick and full must report the same metric unit and direction")

    after_baseline = _git_state(cfg.work)
    if framework_before is not None and _git_state(framework) != framework_before:
        raise SystemExit("generated harness changed the auto-gpu-kernel source checkout")
    if not brief_path.is_file() or brief_path.read_bytes() != brief:
        raise SystemExit("generated harness changed the user's task brief")
    if after_baseline != before:
        raise SystemExit(
            "generated harness changed the target repo while running pristine baselines.\n"
            f"current status:\n{after_baseline[1] or '(clean)'}"
        )

    prepared = taskmod.mark_prepared(cfg, [result.native for result in baselines])
    _commit_generated_harness(cfg)
    print(f"\nHarness ready: {prepared}  rev={generated_rev}")
    return builder.spent


def cmd_run(args) -> int:
    project = Path(args.project).resolve()
    if not (project / "config.toml").exists():
        raise SystemExit(f"no config.toml in {project} — run `kopt init` first")

    cfg = bench_config.load(project)
    setup_spent = _prepare_task(cfg, args) if isinstance(cfg, TaskConfig) else 0.0
    remaining_budget = args.budget
    if remaining_budget is not None:
        remaining_budget = max(0.0, remaining_budget - setup_spent)

    loop = Loop(
        project=project,
        prompt=args.prompt,
        max_iterations=args.iterations,
        budget=remaining_budget,
        max_time=args.max_time,
        timeout=args.timeout,
        model=args.model,
        thinking=args.thinking,
        fresh=args.fresh,
    )
    history = loop.run()
    done = sum(1 for i in history if not i.stalled)
    total_spent = setup_spent + loop.spent
    print(f"\n{len(history)} optimization iterations | {done} produced experiments"
          f" | ${total_spent:.4f}")
    return 0


def cmd_watch(args) -> int:
    project = Path(args.project).resolve()
    if args.list:
        runs = list_run_logs(project)
        if not runs:
            print("no runs recorded")
        for r in runs:
            print(f"{r.stem}  {r.stat().st_size:>9,} B")
        return 0
    serve(project, port=args.port, run=args.run, host=args.host)
    return 0


def main() -> int:
    # Runs are long and usually backgrounded or piped, where Python block-buffers
    # stdout — the log stays empty for minutes and looks hung. Flush per line.
    sys.stdout.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(prog="kopt")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="scaffold a project from a definition JSON")
    i.add_argument("project", help="directory to create")
    i.add_argument("definition", help="path to the definition JSON from the trace set")
    i.add_argument("--language", default="triton", choices=languages())
    i.add_argument("--backend", default="modal", choices=("local", "modal", "fal"))
    i.add_argument("--gpu", default="B200")
    i.add_argument("--force", action="store_true", help="overwrite a non-empty directory")
    i.set_defaults(func=cmd_init)

    t = sub.add_parser("init-task", help="scaffold an isolated project from a task brief")
    t.add_argument("project", help="directory to create")
    t.add_argument("taskspec", help="path to the task brief TOML (becomes config.toml)")
    t.add_argument("--force", action="store_true", help="overwrite a non-empty directory")
    t.set_defaults(func=cmd_init_task)

    r = sub.add_parser("run", help="run the optimization loop")
    r.add_argument("project", nargs="?", default=".")
    r.add_argument("-n", "--iterations", type=int, default=10)
    r.add_argument("--budget", type=float, help="stop once this much USD is spent")
    r.add_argument(
        "--max-time",
        help="omp session lifetime, e.g. 20m; loop reconnects when it expires",
    )
    r.add_argument("--timeout", type=float, default=3600.0, help="seconds per iteration")
    r.add_argument("--model", help="omp model (fuzzy: 'opus', 'claude-sonnet-4-5')")
    r.add_argument(
        "--thinking",
        choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"),
        help="omp thinking level",
    )
    r.add_argument("--fresh", action="store_true",
                   help="new omp session each iteration (default: persist one)")
    r.add_argument("--prompt", default=DEFAULT_PROMPT)
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("watch", help="live web view of a run")
    w.add_argument("project", nargs="?", default=".")
    w.add_argument("-p", "--port", type=int, default=8765)
    w.add_argument("--host", default="127.0.0.1",
                   help="bind address (0.0.0.0 to expose on the network)")
    w.add_argument("--run", help="run to show (name or path); default newest")
    w.add_argument("--list", action="store_true", help="list recorded runs and exit")
    w.set_defaults(func=cmd_watch)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
