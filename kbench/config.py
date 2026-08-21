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
class TaskConfig:
    """Human-authored intent for optimizing an arbitrary repository."""

    root: Path
    # [task]
    name: str
    objective: str
    measure: str
    validate: str
    hints: str
    workdir: str
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

    backend: str = "local"  # provenance label; task mode always runs locally

    @property
    def work(self) -> Path:
        return self.root / self.workdir


def _load_task(root: Path, raw: dict) -> TaskConfig:
    task = raw["task"]
    repo = task.get("repo", {})
    hw = task.get("hardware", {})
    workdir = task.get("workdir", "repo")
    resolved_work = (root / workdir).resolve()
    if resolved_work == root.resolve() or not resolved_work.is_relative_to(root.resolve()):
        raise SystemExit("config.toml: task.workdir must be a child of the task project")
    objective = task.get("objective", task.get("description", "")).strip()
    measure = task.get("measure", task.get("metric", "")).strip()
    validate = task.get("validate", "").strip()
    missing = [
        name
        for name, value in (
            ("task.name", task.get("name")),
            ("task.objective", objective),
            ("task.measure", measure),
            ("task.validate", validate),
            ("task.repo.url", repo.get("url")),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"config.toml: missing {', '.join(missing)}")

    return TaskConfig(
        root=root,
        name=task["name"],
        objective=objective,
        measure=measure,
        validate=validate,
        hints=task.get("hints", "").strip(),
        workdir=workdir,
        repo_url=repo.get("url", ""),
        branch=repo.get("branch", ""),
        base=repo.get("base", "main"),
        gpus=int(hw.get("gpus", 1)),
        gpu=hw.get("gpu", "GPU"),
        env={str(k): str(v) for k, v in task.get("env", {}).items()},
        path_prepend=task.get("path_prepend", ""),
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


def load(root: Path | None = None) -> Config | TaskConfig:
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
