# Task optimization project

Autonomous performance optimization of a real repository. The target repo, its bench
commands, the hardware, and the correctness contract all come from `config.toml` — read it
first. The full task description is in §This task below.

## Non-negotiable rules

- **Absolute seconds only.** Speedup ratios compound noise; every claim is an absolute
  measurement from `kbench`, tied to the git rev it measured.
- **One optimization per iteration.** Coupled changes misattribute wins. For small deltas
  (<2%), confirm with `kbench ab --a <git-ref-of-prev-best>` (back-to-back on this machine).
- **Benchmark through `kbench`.** Never hand-roll the final benchmark; ad-hoc scripts are
  fine for *investigation*, but every logged number comes from `kbench bench`.
- **Correctness is the golden artifact.** A run passes only if the pinned verifier exits 0
  and its artifact matches the golden byte-for-byte-equal JSON. There is no "close enough";
  a MISMATCH is a failed experiment no matter how fast it ran.
- **Log every experiment** via `/skill:log-experiment`, including failures.
- **One optimization per turn, then stop.** After you have logged the experiment, end your
  turn. A supervisor re-invokes you immediately with fresh context — you are not ending the
  optimization, only this step of it.
- **No benchmark gaming.** Never touch the pinned verifier copies (anything under
  `.kbench/` in the workdir) or the `golden/` dir. No memoizing outputs across forward
  calls, no input-pointer caching, no special-casing the verifier's frozen inputs, no
  detecting "am I being benchmarked". Every optimization must hold for arbitrary real
  requests, not just the pinned one.
- **No web access.** You work from the repo and local files only. Do not search the web
  or read remote URLs.
- **Don't ask the user anything.** You are autonomous; the user will not answer.

## Skills

| Skill | Purpose |
|---|---|
| `/skill:optimize` | Main loop — one optimization, benchmarked and logged |
| `/skill:log-experiment` | Snapshot the change + write `result.md` + update the index + push |

## Benchmarking

```bash
kbench bench --quick        # few-step run: correctness gate + smoke timing (~1-2 min)
kbench bench --mode perf    # clean repeated-forward timing (~2-3 min)
kbench bench --mode full    # the real full generation — lock in final numbers (~5-8 min)
kbench ab --a <git-ref>     # back-to-back A/B against a past commit, same machine
```

**Capture the output**: `kbench bench --quick 2>&1 | tee bench.log` — `/skill:log-experiment`
copies `bench.log` into the experiment folder, so a run you didn't tee is a run you can't log.

`kbench` copies the pinned verifier into the workdir, runs the mode's command with all GPUs
visible, streams its output, checks the artifact against the golden, and appends to the
`.kopt/bench.jsonl` timeline. Exit 0 = passed.

After a benchmark, report: pass/fail, golden status, the metric in seconds, per-forward
ms stats (min/mean/median/max), and the workdir rev it measured.

## Repo layout

- `config.toml` — the task spec: repo, bench modes, hardware, correctness contract
- `<workdir>/` — the target repo working tree (see §This task); this is what you edit
- `verify/`, `golden/` — pinned verifier + golden artifacts. **Read-only.**
- `experiments/exp_N/` — per-experiment: `plan.md?`, change snapshot, `result.md`, `bench.log`
- `experiments/summary.md` — master index, one row per experiment
- `experiments/LESSONS.md` — durable cross-experiment findings

## Git

Two repos, two roles:
- **The target repo** (`<workdir>/`): commit after every logged experiment that changed it —
  wins *and* reverts — with a message naming the experiment (e.g. `exp_12: fuse rope+quant,
  287.4s`), and **push to the work branch** named in `config.toml`. Never force-push.
  Never commit into `.kbench/`.
- **The project root**: commit `experiments/` after each `/skill:log-experiment`.
