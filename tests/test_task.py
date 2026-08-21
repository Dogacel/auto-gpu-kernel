from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kbench import adapters
from kbench.adapters.base import BenchmarkAdapter, Measurement, RunRequest
from kbench.adapters.flashinfer import FlashInferAdapter
from kbench.adapters.generated import GeneratedTaskAdapter
from kbench.cli import _request
from kbench.config import Config, TaskConfig, load
from kbench.results import WorkloadResult
from kbench.task import harness_rev, run_ab, run_mode, workdir_rev
from kopt.cli import _git_state, _prepare_task
from kopt.init import init_task


VALIDATE = """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--mode", choices=("quick", "full"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
passed = (args.repo / "answer.txt").read_text().strip() == "42"
args.output.write_text(json.dumps({"passed": passed, "details": args.mode}))
raise SystemExit(0 if passed else 1)
"""


BENCHMARK = """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--mode", choices=("quick", "full"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
value = float((args.repo / "metric.txt").read_text())
samples = [value] if args.mode == "quick" else [value, value, value]
args.output.write_text(json.dumps({
    "value": value,
    "unit": "seconds",
    "lower_is_better": True,
    "samples": samples,
}))
"""


def write_harness(root: Path) -> None:
    harness = root / "harness"
    harness.mkdir(exist_ok=True)
    (harness / "validate.py").write_text(VALIDATE)
    (harness / "benchmark.py").write_text(BENCHMARK)
    (harness / "README.md").write_text("Test quick/full harness.\n")


class TaskHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work = self.root / "repo"
        self.work.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.work, check=True)
        (self.work / "answer.txt").write_text("42\n")
        (self.work / "metric.txt").write_text("2.0\n")
        subprocess.run(["git", "add", "-A"], cwd=self.work, check=True)
        subprocess.run(
            [
                "git", "-c", "user.email=test@localhost", "-c", "user.name=test",
                "commit", "-q", "-m", "baseline",
            ],
            cwd=self.work,
            check=True,
        )
        (self.root / "config.toml").write_text(
            """
[task]
name = "demo"
workdir = "repo"
objective = "Make the demo faster."
measure = "Measure elapsed seconds; lower is better."
validate = "The result must remain exactly 42."
hints = "Keep it simple."

[task.repo]
url = "https://example.invalid/demo.git"
base = "main"

[task.hardware]
gpus = 1
gpu = "test-gpu"
""".lstrip()
        )
        write_harness(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_minimal_brief_loads_without_user_bench_commands(self) -> None:
        cfg = load(self.root)
        self.assertIsInstance(cfg, TaskConfig)
        assert isinstance(cfg, TaskConfig)
        self.assertEqual(cfg.objective, "Make the demo faster.")
        self.assertEqual(cfg.measure, "Measure elapsed seconds; lower is better.")

    def test_workdir_cannot_escape_the_task_project(self) -> None:
        text = (self.root / "config.toml").read_text().replace(
            'workdir = "repo"', 'workdir = "../auto-gpu-kernel"'
        )
        (self.root / "config.toml").write_text(text)
        with self.assertRaisesRegex(SystemExit, "must be a child"):
            load(self.root)

    def test_kbench_runs_validation_then_benchmark(self) -> None:
        cfg = load(self.root)
        assert isinstance(cfg, TaskConfig)
        result = run_mode(cfg, "quick")
        self.assertTrue(result.passed)
        self.assertEqual(result.validation_status, "pass")
        self.assertEqual(result.value, 2.0)
        self.assertEqual(result.unit, "seconds")
        self.assertEqual(result.samples, [2.0])
        self.assertTrue((Path(result.out_dir) / "validation.json").exists())
        self.assertTrue((Path(result.out_dir) / "benchmark.json").exists())

    def test_failed_validation_prevents_benchmark(self) -> None:
        cfg = load(self.root)
        assert isinstance(cfg, TaskConfig)
        (cfg.work / "answer.txt").write_text("wrong\n")
        result = run_mode(cfg, "full")
        self.assertFalse(result.passed)
        self.assertIn("validate.py exited", result.validation_status)
        self.assertFalse((Path(result.out_dir) / "benchmark.json").exists())

    def test_kbench_rejects_an_invalid_benchmark_contract(self) -> None:
        cfg = load(self.root)
        assert isinstance(cfg, TaskConfig)
        benchmark = self.root / "harness" / "benchmark.py"
        benchmark.write_text(BENCHMARK.replace('"unit": "seconds",\n', ""))
        result = run_mode(cfg, "quick")
        self.assertFalse(result.passed)
        self.assertIn("non-empty `unit`", result.error)

    def test_kbench_rejects_an_adapter_that_mutates_its_inputs(self) -> None:
        cfg = load(self.root)
        assert isinstance(cfg, TaskConfig)
        validate = self.root / "harness" / "validate.py"
        validate.write_text(
            VALIDATE.replace(
                "passed = (args.repo",
                '(Path(__file__).parent / "benchmark.py").write_text("changed")\n'
                "passed = (args.repo",
            )
        )
        result = run_mode(cfg, "quick")
        self.assertFalse(result.passed)
        self.assertIn("harness changed", result.error)
        self.assertFalse((Path(result.out_dir) / "benchmark.json").exists())

    def test_harness_edits_are_allowed_and_change_provenance(self) -> None:
        cfg = load(self.root)
        assert isinstance(cfg, TaskConfig)
        before = harness_rev(cfg)
        benchmark = self.root / "harness" / "benchmark.py"
        benchmark.write_text(benchmark.read_text() + "\n# revised harness\n")
        after = harness_rev(cfg)
        self.assertNotEqual(before, after)
        self.assertEqual(run_mode(cfg, "quick").harness_rev, after)

    def test_dirty_provenance_fingerprints_file_contents(self) -> None:
        tracked = self.work / "metric.txt"
        tracked.write_text("3.0\n")
        tracked_rev = workdir_rev(self.work)
        tracked_state = _git_state(self.work)
        tracked.write_text("4.0\n")
        self.assertNotEqual(tracked_rev, workdir_rev(self.work))
        self.assertNotEqual(tracked_state[2], _git_state(self.work)[2])

        untracked = self.work / "new-code.py"
        untracked.write_text("version = 1\n")
        untracked_rev = workdir_rev(self.work)
        untracked.write_text("version = 2\n")
        self.assertNotEqual(untracked_rev, workdir_rev(self.work))

    def test_quick_flag_selects_kbench_quick_mode(self) -> None:
        request = _request(
            SimpleNamespace(mode=None, quick=True, stride=1, env=None)
        )
        self.assertEqual(request.mode, "quick")

    def test_generated_task_implements_the_common_adapter(self) -> None:
        cfg = load(self.root)
        assert isinstance(cfg, TaskConfig)
        adapter = adapters.get(cfg)
        self.assertIsInstance(adapter, BenchmarkAdapter)
        self.assertIsInstance(adapter, GeneratedTaskAdapter)
        result = adapter.bench(RunRequest(mode="quick"))
        self.assertTrue(result.passed)
        self.assertEqual(result.adapter, "generated-task")
        self.assertEqual(result.value, 2.0)

    def test_custom_adapter_only_needs_to_implement_run(self) -> None:
        class TinyAdapter(BenchmarkAdapter):
            name = "tiny"

            def run(self, candidates, request):
                return {
                    label: Measurement(
                        adapter=self.name,
                        label=label,
                        candidate=selector or "current",
                        mode=request.mode,
                        passed=True,
                        value=7.0,
                        unit="widgets/s",
                        lower_is_better=False,
                        samples=[7.0],
                        validation="pass",
                        provenance="test machine",
                    )
                    for label, selector in candidates.items()
                }

        adapter = TinyAdapter(self.root)
        result = adapter.bench(RunRequest(mode="quick"))
        adapter.record(result)
        self.assertTrue(result.passed)
        self.assertEqual(result.value, 7.0)
        self.assertTrue((self.root / ".kopt" / "bench.jsonl").exists())

    def test_kernel_config_selects_the_flashinfer_adapter(self) -> None:
        kernel_root = self.root / "kernel-project"
        kernel_root.mkdir()
        (kernel_root / "config.toml").write_text(
            """
[kernel]
definition = "demo"
language = "triton"
source_dir = "solution/triton"
entry_point = "solution.py::kernel"

[remote]
backend = "local"
gpu = "test-gpu"
""".lstrip()
        )
        cfg = load(kernel_root)
        self.assertIsInstance(cfg, Config)
        adapter = adapters.get(cfg)
        self.assertIsInstance(adapter, BenchmarkAdapter)
        self.assertIsInstance(adapter, FlashInferAdapter)

        source = cfg.sources / "solution.py"
        source.parent.mkdir(parents=True)
        source.write_text("def kernel(): pass\n")
        baseline = kernel_root / "baseline.py"
        baseline.write_text("def kernel(): return 1\n")

        class FakeBackend:
            def __init__(self):
                self.jobs = []

            def run(self, _cfg, job):
                self.jobs.append(job)
                return {
                    label: [
                        WorkloadResult(
                            workload_id="workload-1",
                            status="passed",
                            latency_ms=1.25,
                        )
                    ]
                    for label in job.sources
                }

        backend = FakeBackend()
        with patch("kbench.adapters.flashinfer.backends.get", return_value=backend):
            result = adapter.bench(RunRequest(mode="quick"))
            a, b = adapter.ab(str(baseline), RunRequest(mode="full"))

        self.assertTrue(result.passed)
        self.assertEqual(result.value, 1.25)
        self.assertEqual(result.unit, "ms")
        self.assertTrue(adapter.comparable(a, b))
        self.assertEqual(set(backend.jobs[-1].sources), {"a", "b"})

    def test_ab_uses_the_same_current_harness(self) -> None:
        cfg = load(self.root)
        assert isinstance(cfg, TaskConfig)
        baseline = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cfg.work,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        (cfg.work / "metric.txt").write_text("1.5\n")
        a, b = run_ab(cfg, baseline, "full")
        self.assertTrue(a.passed and b.passed)
        self.assertEqual(a.harness_rev, b.harness_rev)
        self.assertEqual(a.value, 2.0)
        self.assertEqual(b.value, 1.5)

    def test_init_task_creates_an_isolated_builder_project(self) -> None:
        spec = self.root / "brief.toml"
        spec.write_text(
            f"""
[task]
name = "scaffold-demo"
objective = "Make it faster."
measure = "Measure wall time; lower is better."
validate = "Preserve the answer."

[task.repo]
url = {json.dumps(str(self.work))}

[task.hardware]
gpus = 1
gpu = "test-gpu"
""".lstrip()
        )
        project = self.root / "scaffold"
        init_task(project, spec)

        self.assertTrue((project / "repo" / ".git").exists())
        skill = project / ".omp" / "skills" / "build-harness" / "SKILL.md"
        self.assertTrue(skill.exists())
        agents = (project / ".omp" / "AGENTS.md").read_text()
        self.assertIn("target repository is read-only", agents)
        self.assertIn("Kbench owns the experiment lifecycle", agents)
        self.assertFalse((project / "harness").exists())
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project,
            check=True, capture_output=True, text=True,
        ).stdout
        self.assertEqual(status, "")

    def test_first_run_builds_both_baselines_and_commits(self) -> None:
        spec = self.root / "automatic.toml"
        spec.write_text(
            f"""
[task]
name = "automatic-demo"
objective = "Make it faster."
measure = "Measure wall time; lower is better."
validate = "Preserve the answer."

[task.repo]
url = {json.dumps(str(self.work))}

[task.hardware]
gpus = 1
gpu = "test-gpu"
""".lstrip()
        )
        project = self.root / "automatic"
        init_task(project, spec)
        cfg = load(project)
        assert isinstance(cfg, TaskConfig)

        class FakeBuilder:
            spent = 0.25

            def __init__(self, **kwargs):
                self.project = kwargs["project"]

            def run(self):
                write_harness(self.project)
                return []

        args = SimpleNamespace(
            timeout=10.0, max_time=None, model="test", thinking="low",
        )
        with patch("kopt.cli.Loop", FakeBuilder):
            spent = _prepare_task(cfg, args)

        self.assertEqual(spent, 0.25)
        prepared = json.loads((project / "harness" / "prepared.json").read_text())
        self.assertEqual(set(prepared["baseline"]), {"quick", "full"})
        subject = subprocess.run(
            ["git", "log", "-1", "--format=%s"], cwd=project,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(subject, "build generated harness")
        target_status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cfg.work,
            check=True, capture_output=True, text=True,
        ).stdout
        self.assertEqual(target_status, "")


if __name__ == "__main__":
    unittest.main()
