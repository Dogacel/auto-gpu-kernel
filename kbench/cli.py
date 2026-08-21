"""kbench bench | ab"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kbench import adapters, config
from kbench.adapters.base import RunRequest


def _env(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        if key:
            out[key.strip()] = value.strip()
    return out


def _request(args) -> RunRequest:
    mode = getattr(args, "mode", None)
    if not mode:
        mode = "quick" if args.quick else "stride" if args.stride > 1 else "full"
    return RunRequest(
        mode=mode,
        stride=args.stride,
        env=_env(args.env),
    )


def cmd_bench(args) -> int:
    cfg = config.load()
    adapter = adapters.get(cfg)
    result = adapter.bench(_request(args))
    adapter.record(result)
    adapter.print_result(result)

    if args.json:
        Path(args.json).write_text(json.dumps(adapter.serialize(result), indent=2))
        print(f"\nwrote {args.json}")
    return 0 if result.passed else 1


def cmd_ab(args) -> int:
    cfg = config.load()
    adapter = adapters.get(cfg)
    a, b = adapter.ab(args.a, _request(args))
    adapter.print_ab(a, b)
    return 0 if adapter.comparable(a, b) else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="kbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--quick", action="store_true", help="use the adapter's quick path")
        p.add_argument(
            "--mode", choices=("quick", "full"), help="benchmark adapter mode"
        )
        p.add_argument(
            "--stride", type=int, default=1,
            help="FlashInfer adapter: sample every Nth workload",
        )
        p.add_argument("--env", action="append", help="K=V passed to adapter execution")

    b = sub.add_parser("bench", help="benchmark the current candidate")
    add_common(b)
    b.add_argument("--json", help="write results to this path")
    b.set_defaults(func=cmd_bench)

    a = sub.add_parser("ab", help="paired A/B against another candidate")
    add_common(a)
    a.add_argument(
        "--a", required=True,
        help="baseline selector: FlashInfer source file or generated-task git ref",
    )
    a.set_defaults(func=cmd_ab)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
