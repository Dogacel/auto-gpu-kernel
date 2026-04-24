# Experiment 54 — 2026-04-17

**Description:** Per `plan.md`, attempted adaptive NUM_SPLITS dispatch to recover
the `4c46a94b` +14% regression logged in exp_51. Two plan variants tested:
- **Action 1 (primary):** `valid_max` host-side dispatch via
  `(sparse_indices >= 0).sum(-1).amax().item()`, route `valid_max > 1024 → NS=16,
  else NS=8`.
- **Action 2 (fallback):** T-based rule per plan's assumption that 4c46a94b is T=6,
  `num_tokens >= 7 → NS=16, else NS=8`.

Also extended `_counter_cache` key to `(device, num_tokens, num_splits)` to
preserve monotonic-barrier correctness under dispatch-switched NUM_SPLITS.

## Results
- Pass: 23/23 full (variant A), 2/2 quick (variant B — but catastrophic latency)
- Mode: variant A ran A/B stride-2 × 2 + full. Variant B aborted after quick.

### Variant A (Action 2, T-based dispatch) — neutral
| Run | Paired n | B wins | Mean Δ |
|---|---|---|---|
| 1 | 12 | 9/12 | -0.0000 ms |
| 2 | 12 | 8/12 | -0.0000 ms |

Small directional B-wins on most workloads (~0.5-1% faster), but **mean Δ
indistinguishable from zero**. 4c46a94b improvement: 0.79% → 0.51% (noise-floor).
The plan's critical premise failed: **`4c46a94b` is T=8, valid_max=1002** per
`workload_profile.md` line 89 — NOT T=6 as plan.md claimed. With my T≥7 → NS=16
rule, 4c46a94b STILL routes to NS=16 (same as exp_51) → no effect on the target
workload. The rule only switches T=6 workloads (3 of 23, not sampled by stride-2)
to NS=8; unclear if that's helpful.

### Variant B (Action 1, `.item()` dispatch) — catastrophic
2207f0fd went **0.013 ms → 0.103 ms (~9× regression)**. Host-side `.item()`
sync stall is the smoking gun. Just 4 chained pytorch ops (`>= 0`, `.sum`,
`.amax`, `.item`) + sync + memcpy added ~100 µs to a ~12 µs kernel. Confirms
LESSON-20 to the letter: "num_valid not cheaply available from host — .item()
sync costs more than saving."

The `.item()` adds three kernel launches on the critical path PLUS a
device→host sync+memcpy. Even on a cupti-timed benchmark, the sync point stalls
the submission queue so subsequent kernel work shows up as extended duration.

## Learnings

**1. Plan premise error (not caught pre-implementation).** Plan.md line 77-79
asserted "4c46a94b is the only T=6 in stride-2" — factually wrong.
`workload_profile.md` line 89 shows `4c46a94b: T=8, valid_max=1002`. The
valid_max-based dispatch rule (Action 1) is mechanistically correct for that
workload, but its host-side cost makes it infeasible. The T-based Action 2
fallback is ineffective because T alone can't separate 4c46a94b (T=8) from
other T=8 workloads.

**2. `.item()` cost is enormous** — ~90 µs added to a 12 µs kernel. This is
7.5× the kernel runtime. The cupti harness DOES measure this because the sync
serializes subsequent kernel timing (GPU queue stalls).

**3. Adaptive dispatch needs device-side branching, not host.** To use valid_max
to pick NUM_SPLITS, the decision must stay on-device. Options:
   - Always launch max NUM_SPLITS (=16), have each CTA self-skip if its assigned
     indices are all `-1` (nearly already in place via `max_bn=0` early-exit).
     Skipping CTAs still cost launch+barrier overhead though.
   - Embed the dispatch in a single kernel that branches on a runtime-loaded
     valid_count to vary the work pattern. Breaks the constexpr NUM_SPLITS model.
   - Use torch compile/graph to fuse valid_max into the kernel launch —
     explicitly banned by CLAUDE.md.

**4. 4c46a94b's +14% regression from exp_51 is LIKELY UNRECOVERABLE under the
current kernel structure** without device-side adaptive branching. Net impact is
small (1.6 µs / call on a single workload). Moving on to other axes.

## Decision
Revert to exp_51 (new kernel file in this folder is byte-identical to exp_51).
Plan's adaptive-dispatch axis is **closed**: Action 1 too costly; Action 2
misdiagnosed. Leave 4c46a94b regression as a known, bounded cost (~1.6 µs).

Next likely axes (from exp_53/result.md's close-out and profile 2.25 µs
split-phase gap):
- Warp-level intra-CTA combine parallelism (structural — not a drop-in).
- Split-phase micro-tuning (2.25 µs compute gap vs HBM floor).
- DSA_PROFILE instrumentation to break down combine vs split in the fused kernel.
