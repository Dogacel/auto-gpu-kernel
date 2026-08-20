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


def _tail(path: Path):
    """Yield JSON records, then follow the file as it grows."""
    pos = 0
    while True:
        if path.exists():
            with path.open() as fh:
                fh.seek(pos)
                for line in fh:
                    if not line.endswith("\n"):
                        break  # partial write; re-read next pass
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
                pos = fh.tell()
        time.sleep(0.25)


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


def serve(project: Path, port: int = 8765, run: str | None = None) -> None:
    project = Path(project).resolve()
    experiments = (project / "experiments").resolve()
    pinned_log = resolve(project, run)
    pinned = pinned_log is not None

    def current_log() -> Path | None:
        # Without an explicit run, keep resolving so `kopt watch` can start first.
        return pinned_log if pinned else latest_run_log(project)

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
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while (path := current_log()) is None:
                        time.sleep(0.5)
                    for record in _tail(path):
                        self.wfile.write(f"data: {json.dumps(record)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # EventSource reconnects on its own
                return

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

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    print(f"http://127.0.0.1:{port}")
    if pinned_log is not None:
        print(f"  log {pinned_log.name}"
              + ("" if pinned else f"  (newest of {len(list_run_logs(project))})"))
    else:
        print("  no run yet — the log fills in when `kopt run` starts")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
