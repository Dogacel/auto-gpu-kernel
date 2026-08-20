"""config.toml -> typed config. One source of truth for image, GPU, and paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class ImageSpec:
    """Declarative image. Modal replays it as builder calls; fal renders a Dockerfile."""

    base: str
    apt: tuple[str, ...] = ()
    pip: tuple[str, ...] = ()
    run: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def to_dockerfile(self) -> str:
        lines = [f"FROM {self.base}"]
        if self.apt:
            lines.append(
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                + " ".join(self.apt)
            )
        if self.pip:
            lines.append("RUN pip install " + " ".join(f'"{p}"' for p in self.pip))
        lines += [f"RUN {c}" for c in self.run]
        lines += [f"ENV {k}={v}" for k, v in self.env.items()]
        return "\n".join(lines)


@dataclass(frozen=True)
class BenchMode:
    """One named benchmark mode of a task project (e.g. quick / perf / full)."""

    name: str
    cmd: str
    """Shell command run with cwd=<workdir>. Placeholders: {out} = artifact dir."""
    timeout_s: int = 1800
    golden: str = ""
    """Project-relative golden JSON; compared for exact equality against `artifact`."""
    artifact: str = ""
    """Filename the command writes into {out} (e.g. ledger.json)."""
    metric_regex: str = ""
    """Regex over stdout with a named group `value` (float, seconds)."""
    step_regex: str = ""
    """Regex over stdout with named group `ms`; findall gives per-step latencies."""
    metric_json: tuple[str, ...] = ()
    """(artifact_name, dotted.path) — read the metric from an artifact JSON instead."""


@dataclass(frozen=True)
class PinnedFile:
    """A harness-owned file copied fresh into the workdir before every bench.

    Pinning is what keeps the measurement trustworthy: the verifier the benchmark
    runs is the project's copy, not whatever currently sits in the (agent-edited)
    working tree.
    """

    src: str  # project-relative
    dst: str  # workdir-relative
    from_repo: str = ""  # repo-relative origin, snapshotted once by `kopt init-task`


@dataclass(frozen=True)
class TaskConfig:
    """A generic optimization task: a git repo, bench commands, and a metric."""

    root: Path
    # [task]
    name: str
    description: str
    workdir: str
    metric: str
    lower_is_better: bool
    # [task.repo]
    repo_url: str
    branch: str
    base: str
    # [task.hardware]
    gpus: int
    gpu: str
    # [task.env]
    env: dict[str, str]
    path_prepend: str
    # [[task.pinned]]
    pinned: tuple[PinnedFile, ...]
    # [task.bench.*]
    modes: dict[str, BenchMode]
    default_mode: str

    backend: str = "local"  # provenance label; task mode always runs locally

    @property
    def work(self) -> Path:
        return self.root / self.workdir


def _load_task(root: Path, raw: dict) -> TaskConfig:
    task = raw["task"]
    repo = task.get("repo", {})
    hw = task.get("hardware", {})
    bench = task.get("bench", {})
    if not bench:
        raise SystemExit("config.toml: [task] needs at least one [task.bench.<mode>]")

    modes = {}
    for name, m in bench.items():
        mj = m.get("metric_json", ())
        modes[name] = BenchMode(
            name=name,
            cmd=m["cmd"],
            timeout_s=int(m.get("timeout_s", 1800)),
            golden=m.get("golden", ""),
            artifact=m.get("artifact", ""),
            metric_regex=m.get("metric_regex", ""),
            step_regex=m.get("step_regex", ""),
            metric_json=tuple(mj),
        )

    return TaskConfig(
        root=root,
        name=task["name"],
        description=task.get("description", ""),
        workdir=task.get("workdir", "repo"),
        metric=task.get("metric", "seconds (lower is better)"),
        lower_is_better=task.get("lower_is_better", True),
        repo_url=repo.get("url", ""),
        branch=repo.get("branch", ""),
        base=repo.get("base", "main"),
        gpus=int(hw.get("gpus", 1)),
        gpu=hw.get("gpu", "GPU"),
        env={str(k): str(v) for k, v in task.get("env", {}).items()},
        path_prepend=task.get("path_prepend", ""),
        pinned=tuple(
            PinnedFile(src=p["src"], dst=p["dst"], from_repo=p.get("from_repo", ""))
            for p in task.get("pinned", [])
        ),
        modes=modes,
        default_mode=task.get("default_mode", "full" if "full" in modes else next(iter(modes))),
    )


@dataclass(frozen=True)
class Config:
    root: Path
    # [kernel]
    definition: str
    language: str
    source_dir: str
    entry_point: str
    # [remote]
    backend: str
    gpu: str
    gpu_count: int
    timeout_s: int
    image: ImageSpec
    # [remote.data]
    data_path: str  # where the trace set is mounted inside the container
    modal_volume: str
    local_path: str
    # [bench]
    bench: dict

    @property
    def sources(self) -> Path:
        return self.root / "solution" / self.source_dir

    @property
    def kernel(self) -> Path:
        """The file being optimized, per entry_point."""
        return self.sources / self.entry_point.split("::")[0]


def load(root: Path | None = None) -> Config:
    root = Path(root or os.environ.get("KBENCH_ROOT") or Path.cwd()).resolve()
    # Walk up so `kbench` works from anywhere inside the project (e.g. the task workdir).
    for candidate in (root, *root.parents):
        if (candidate / "config.toml").exists():
            root = candidate
            break
    path = root / "config.toml"
    if not path.exists():
        raise SystemExit(f"no config.toml in {root} (set KBENCH_ROOT or cd to the project)")

    raw = tomllib.loads(path.read_text())
    if "task" in raw:
        return _load_task(root, raw)
    if "kernel" not in raw:
        raise SystemExit(f"{path}: missing [kernel] or [task] section")
    kernel = raw["kernel"]
    remote = raw.get("remote", {})
    img = remote.get("image", {})
    data = remote.get("data", {})

    return Config(
        root=root,
        definition=kernel["definition"],
        language=kernel.get("language", "triton"),
        source_dir=kernel.get("source_dir", kernel.get("language", "triton")),
        entry_point=kernel["entry_point"],
        backend=os.environ.get("KBENCH_BACKEND") or remote.get("backend", "local"),
        gpu=remote.get("gpu", "B200"),
        gpu_count=remote.get("gpu_count", 1),
        timeout_s=remote.get("timeout_s", 1800),
        image=ImageSpec(
            base=img.get("base", ""),
            apt=tuple(img.get("apt", ())),
            pip=tuple(img.get("pip", ())),
            run=tuple(img.get("run", ())),
            env=dict(img.get("env", {})),
        ),
        data_path=data.get("path", "/data"),
        modal_volume=data.get("modal_volume", "flashinfer-trace"),
        local_path=os.path.expanduser(data.get("local_path", "~/flashinfer-trace")),
        bench=raw.get("bench", {}),
    )
