"""kbench bench | ab"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kbench import backends, config
from kbench import task as taskmod
from kbench.config import TaskConfig
from kbench.harness import collect_sources
from kbench.history import record
from kbench.job import Job
from kbench.results import BenchResult, print_ab, print_results


def _job(cfg, sources: dict[str, dict[str, str]], args) -> Job:
    bench = cfg.bench
    return Job(
        definition=cfg.definition,
        language=cfg.language,
        entry_point=cfg.entry_point,
        sources=sources,
        data_path=cfg.data_path,
        mode="quick" if args.quick else "stride" if args.stride > 1 else "full",
        stride=args.stride,
        warmup_runs=bench.get("warmup_runs", 3),
        iterations=bench.get("iterations", 100),
        num_trials=bench.get("num_trials", 5),
        profile_baseline=bench.get("profile_baseline", False),
        env=_env(args.env),
    )


def _run(cfg, job: Job) -> dict[str, BenchResult]:
    results = backends.get(cfg.backend).run(cfg, job)
    return {
        label: BenchResult(
            definition=cfg.definition,
            backend=cfg.backend,
            gpu=cfg.gpu,
            workloads={w.workload_id: w for w in workloads},
        )
        for label, workloads in results.items()
    }


def _env(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        if key:
            out[key.strip()] = value.strip()
    return out


def _task_mode(cfg: TaskConfig, args):
    """Resolve which [task.bench.<mode>] the flags name."""
    name = getattr(args, "mode", None)
    if not name:
        name = "quick" if args.quick else cfg.default_mode
    if name not in cfg.modes:
        raise SystemExit(f"unknown bench mode {name!r}; have: {', '.join(cfg.modes)}")
    return cfg.modes[name]


def cmd_bench_task(cfg: TaskConfig, args) -> int:
    mode = _task_mode(cfg, args)
    result = taskmod.run_mode(
        cfg, mode,
        capture_golden=getattr(args, "capture_golden", False),
        extra_env=_env(args.env),
    )
    taskmod.print_result(cfg, result)
    taskmod.record(cfg, result)
    if args.json:
        from dataclasses import asdict

        Path(args.json).write_text(json.dumps(asdict(result), indent=2))
        print(f"\nwrote {args.json}")
    return 0 if result.passed else 1


def cmd_ab_task(cfg: TaskConfig, args) -> int:
    mode = _task_mode(cfg, args)
    a, b = taskmod.run_ab(cfg, args.a, mode)
    taskmod.print_ab(cfg, a, b, args.a)
    return 0


def cmd_bench(args) -> int:
    cfg = config.load()
    if isinstance(cfg, TaskConfig):
        return cmd_bench_task(cfg, args)
    print(f"{cfg.kernel.relative_to(cfg.root)} -> {cfg.backend}/{cfg.gpu}")

    job = _job(cfg, {"main": collect_sources(cfg)}, args)
    result = _run(cfg, job)["main"]
    record(cfg, result, job.mode if job.mode != "stride" else f"stride{job.stride}")
    print_results(result)

    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nwrote {args.json}")
    return 0 if result.n_passed == len(result.workloads) else 1


def cmd_ab(args) -> int:
    """Both candidates run back-to-back in one container: one VM, one GPU, paired delta."""
    cfg = config.load()
    if isinstance(cfg, TaskConfig):
        return cmd_ab_task(cfg, args)
    a_path = Path(args.a).resolve()
    print(f"A = {a_path}\nB = {cfg.kernel}\n")

    sources = {
        "a": collect_sources(cfg, override=a_path),
        "b": collect_sources(cfg),
    }
    results = _run(cfg, _job(cfg, sources, args))
    print_ab(results["a"], results["b"], str(a_path), str(cfg.kernel))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="kbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--quick", action="store_true", help="smallest + largest workload only"
                       " (task mode: the `quick` bench mode)")
        p.add_argument("--mode", help="task mode: named [task.bench.<mode>] to run")
        p.add_argument("--stride", type=int, default=1, help="sample every Nth workload")
        p.add_argument("--env", action="append", help="K=V passed into the container")

    b = sub.add_parser("bench", help="benchmark the current kernel")
    add_common(b)
    b.add_argument("--json", help="write results to this path")
    b.add_argument("--capture-golden", action="store_true",
                   help="task mode: store this run's artifact as the golden reference")
    b.set_defaults(func=cmd_bench)

    a = sub.add_parser("ab", help="paired A/B against another kernel file (task mode: a git ref)")
    add_common(a)
    a.add_argument("--a", required=True,
                   help="baseline kernel file, or in task mode a git ref/commit")
    a.set_defaults(func=cmd_ab)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
