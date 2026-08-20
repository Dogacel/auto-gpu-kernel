"""The contract between the driving machine and the GPU container.

Flat and primitive-only so it survives whatever each backend uses to serialize
arguments. `kbench` is shipped into the container, so both ends import this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Job:
    definition: str
    """Kernel definition to look up in the trace set."""

    language: str
    entry_point: str
    """e.g. "solution_fused.py::kernel" — file relative to the source dir, and symbol."""

    sources: dict[str, dict[str, str]]
    """label -> {relative_path: file contents}. One label per candidate; A/B sends two."""

    data_path: str
    """Where the trace set is mounted inside the container."""

    mode: str = "full"
    """full | quick | stride"""
    stride: int = 2

    warmup_runs: int = 3
    iterations: int = 100
    num_trials: int = 5
    profile_baseline: bool = False

    env: dict[str, str] = field(default_factory=dict)
    """Set inside the container — shell env does not propagate to remote backends."""

    def select(self, workloads: list) -> list:
        if self.mode == "quick":
            # smallest + largest: catches shape-assumption bugs a single workload hides
            return [workloads[0], workloads[-1]] if len(workloads) >= 2 else workloads[:1]
        if self.mode == "stride":
            return workloads[:: max(1, self.stride)]
        return workloads

    @property
    def kernel_file(self) -> str:
        return self.entry_point.split("::")[0]
