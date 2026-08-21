"""Viewer for a run: live log, optimization chart, and the experiments tree.

Knows nothing about omp or the loop — it reads files. Start it before, during, or
after a run; it replays what already happened, then follows.

Stdlib only: SSE rather than WebSockets keeps this dependency-free, and browsers
reconnect automatically.
"""

from __future__ import annotations

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from kbench.history import kernel_digest
from kbench.history import load as load_history
from kopt import pages
from kopt.record import latest_run_log, list_run_logs, runs_dir

# Only these are browsable, and only under experiments/.
VIEWABLE = {".md", ".py", ".json", ".txt", ".log", ".toml", ".cu", ".cuh", ".jsonl"}
MAX_VIEW_BYTES = 2_000_000


def _tail(path: Path, pos: int = 0):
    """Yield (record, end_pos) for complete lines from pos; (None, pos) idle ticks."""
    while True:
        emitted = False
        if path.exists():
            with path.open("rb") as fh:
                fh.seek(pos)
                while True:
                    line = fh.readline()
                    if not line or not line.endswith(b"\n"):
                        break  # partial write; re-read next pass
                    pos = fh.tell()
                    try:
                        yield json.loads(line), pos
                        emitted = True
                    except json.JSONDecodeError:
                        pass
        if not emitted:
            yield None, pos
        time.sleep(0.25)


# A fresh SSE connection replays at most this much of the log. The logs mirror
# every streaming delta and grow to hundreds of MB; replaying one in full is
# what made EventSource drop and re-replay in a loop.
REPLAY_BYTES = 4 * 1024 * 1024

# Records the log page actually renders — everything else is dead weight on the
# wire (message_update deltas alone are ~90% of the bytes).
def _wanted(rec: dict) -> bool:
    kind = rec.get("kind")
    if kind in ("run_start", "run_end", "reconnect", "iteration"):
        return True
    return kind == "event" and rec.get("type") == "message_end"


DESC = re.compile(r"^\*\*Description:\*\*\s*(.+)$", re.M)


def _experiments(root: Path) -> list[dict]:
    """One entry per exp_N: its description and the hash of its kernel snapshot.

    Hashing the snapshot is what ties an experiment to a measurement — the same
    digest kbench records — so the chart labels points without guessing.
    """
    out = []
    if not root.is_dir():
        return out

    # summary.md's Description column is written as "one phrase" by the
    # log-experiment skill — far better as a chart label than result.md's
    # paragraph, so prefer it and fall back only when a row is missing.
    phrases: dict[str, str] = {}
    index = root / "summary.md"
    if index.exists():
        for line in index.read_text(errors="replace").splitlines():
            cells = [c.strip() for c in line.split("|")]
            if len(cells) > 4 and cells[1].isdigit():
                phrases[f"exp_{cells[1]}"] = cells[3][:70]
    for d in sorted(root.glob("exp_*"), key=lambda p: (len(p.name), p.name)):
        if not d.is_dir():
            continue
        result = d / "result.md"
        desc = phrases.get(d.name, "")
        if not desc and result.exists():
            m = DESC.search(result.read_text(errors="replace"))
            if m:
                desc = " ".join(m.group(1).split())[:90]
        snapshots = [p for p in d.glob("*.py") if p.is_file()]
        out.append({
            "exp": d.name,
            "desc": desc,
            "kernel": kernel_digest(snapshots[0]) if len(snapshots) == 1 else "",
        })
    return out


def resolve(project: Path, run: str | None = None) -> Path | None:
    """Pick which run to show: an explicit path/name, else the newest."""
    if run:
        candidate = Path(run)
        if candidate.exists():
            return candidate
        named = runs_dir(project) / (run if run.endswith(".jsonl") else f"{run}.jsonl")
        if named.exists():
            return named
        raise SystemExit(f"no such run: {run}")
    return latest_run_log(project)


def serve(project: Path, port: int = 8765, run: str | None = None,
          host: str = "127.0.0.1") -> None:
    project = Path(project).resolve()
    experiments = (project / "experiments").resolve()
    pinned_log = resolve(project, run)
    pinned = run is not None

    def current_log() -> Path | None:
        # Without an explicit run, keep resolving so `kopt watch` can start first.
        return pinned_log if pinned else latest_run_log(project)

    def run_by_name(name: str) -> Path | None:
        """?run= comes from the browser: accept run-log *names* only, never paths."""
        name = Path(name).name
        target = runs_dir(project) / (name if name.endswith(".jsonl") else f"{name}.jsonl")
        return target if target.is_file() else None

    def safe(rel: str) -> Path | None:
        """Confine reads to experiments/ — the path comes from the browser."""
        try:
            target = (experiments / rel).resolve()
        except OSError:
            return None
        if not target.is_relative_to(experiments) or not target.is_file():
            return None
        if target.suffix.lower() not in VIEWABLE:
            return None
        return target

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            route = url.path

            if route == "/api/events":
                run_q = (parse_qs(url.query).get("run") or [None])[0]
                if run_q and run_by_name(run_q) is None:
                    return self._send(b"no such run", "text/plain", 404)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while (path := run_by_name(run_q) if run_q else current_log()) is None:
                        time.sleep(0.5)

                    # Resume where the client left off (EventSource sends
                    # Last-Event-ID on reconnect); ids are "<logname>:<pos>" so
                    # a resume never seeks into a different run's file.
                    pos = 0
                    last = self.headers.get("Last-Event-ID", "")
                    name, _, offset = last.rpartition(":")
                    if name == path.name and offset.isdigit():
                        pos = min(int(offset), path.stat().st_size)
                    elif path.stat().st_size > REPLAY_BYTES:
                        # Fresh connect on a huge log: full message detail only
                        # for the tail, but backfill the meta skeleton
                        # (run_start / iteration / run_end rows) from the
                        # skipped region so the whole run's shape still shows.
                        with path.open("rb") as fh:
                            fh.seek(path.stat().st_size - REPLAY_BYTES)
                            fh.readline()  # skip into line alignment
                            pos = fh.tell()
                        meta = (b'"kind": "run_start"', b'"kind": "run_end"',
                                b'"kind": "iteration"', b'"kind": "reconnect"')
                        with path.open("rb") as fh:
                            while fh.tell() < pos and (line := fh.readline()):
                                if not any(m in line for m in meta):
                                    continue
                                try:
                                    rec = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                self.wfile.write(
                                    f"id: {path.name}:{pos}\ndata: {json.dumps(rec)}\n\n".encode())
                            self.wfile.flush()

                    idle = 0.0
                    for record, pos in _tail(path, pos):
                        if record is None:
                            idle += 0.25
                            if idle >= 15:
                                self.wfile.write(b": ping\n\n")  # keep-alive
                                self.wfile.flush()
                                idle = 0.0
                            continue
                        idle = 0.0
                        if not _wanted(record):
                            continue
                        self.wfile.write(
                            f"id: {path.name}:{pos}\ndata: {json.dumps(record)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # EventSource reconnects on its own
                return

            if route == "/api/runs":
                runs = [
                    {"name": p.stem, "size": p.stat().st_size}
                    for p in list_run_logs(project)
                ]
                return self._send(json.dumps(runs).encode(), "application/json")

            if route == "/api/history":
                return self._send(json.dumps(load_history(project)).encode(), "application/json")

            if route == "/api/experiments":
                return self._send(json.dumps(_experiments(experiments)).encode(),
                                  "application/json")

            if route == "/api/tree":
                files = sorted(
                    str(p.relative_to(experiments))
                    for p in experiments.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIEWABLE
                ) if experiments.is_dir() else []
                return self._send(json.dumps(files).encode(), "application/json")

            if route == "/api/file":
                rel = (parse_qs(url.query).get("path") or [""])[0]
                target = safe(rel)
                if target is None:
                    return self._send(b"not viewable", "text/plain", 404)
                return self._send(target.read_bytes()[:MAX_VIEW_BYTES], "text/plain; charset=utf-8")

            page = {
                "/": pages.log_page,
                "/chart": pages.chart_page,
                "/files": pages.files_page,
            }.get(route)
            if page is None:
                return self._send(b"not found", "text/plain", 404)
            return self._send(page().encode(), "text/html; charset=utf-8")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"http://{host}:{port}")
    if pinned_log is not None:
        print(f"  log {pinned_log.name}"
              + ("" if pinned else f"  (newest of {len(list_run_logs(project))})"))
    else:
        print("  no run yet — the log fills in when `kopt run` starts")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
