"""kopt — scaffold a kernel project, then run the optimization loop over it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kopt.init import init, languages
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


def cmd_run(args) -> int:
    project = Path(args.project).resolve()
    if not (project / "config.toml").exists():
        raise SystemExit(f"no config.toml in {project} — run `kopt init` first")

    loop = Loop(
        project=project,
        prompt=args.prompt,
        max_iterations=args.iterations,
        budget=args.budget,
        max_time=args.max_time,
        timeout=args.timeout,
        model=args.model,
        fresh=args.fresh,
    )
    history = loop.run()
    done = sum(1 for i in history if not i.stalled)
    print(f"\n{len(history)} iterations | {done} produced experiments | ${loop.spent:.4f}")
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
    serve(project, port=args.port, run=args.run)
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

    r = sub.add_parser("run", help="run the optimization loop")
    r.add_argument("project", nargs="?", default=".")
    r.add_argument("-n", "--iterations", type=int, default=10)
    r.add_argument("--budget", type=float, help="stop once this much USD is spent")
    r.add_argument("--max-time", help="omp session lifetime, e.g. 20m; loop reconnects when it expires")
    r.add_argument("--timeout", type=float, default=3600.0, help="seconds per iteration")
    r.add_argument("--model", help="omp model (fuzzy: 'opus', 'claude-sonnet-4-5')")
    r.add_argument("--fresh", action="store_true",
                   help="new omp session each iteration (default: persist one)")
    r.add_argument("--prompt", default=DEFAULT_PROMPT)
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("watch", help="live web view of a run")
    w.add_argument("project", nargs="?", default=".")
    w.add_argument("-p", "--port", type=int, default=8765)
    w.add_argument("--run", help="run to show (name or path); default newest")
    w.add_argument("--list", action="store_true", help="list recorded runs and exit")
    w.set_defaults(func=cmd_watch)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
