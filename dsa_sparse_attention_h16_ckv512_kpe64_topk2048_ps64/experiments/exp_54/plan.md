# Plan — exp 54

## Diagnosis

Plateau entered after exp_51 (current best, NUM_SPLITS=16). Three consecutive reverts
(exp_52 dynamic-range combine, exp_53 tree/offline-softmax combine in 3 variants)
deeply closed the **combine-loop reformulation axis** — LESSON-55/56 agree the
static_range + online-softmax pattern is already compiler-ILP-maximized and
cannot be beaten by semantic rewrites. The one concrete, well-understood,
*pre-derived* open lever is the `4c46a94b` regression documented in both
`exp_51/result.md` ("10:1 win/loss ratio, 1.6 µs lost per call") and
`workload_profile.md` lines 84-95 (stride-2 iter-count table), which gives the
exact dispatch rule. Profile's other gap — split_work 2.25 µs above HBM floor —
requires instrumentation before coding and is deferred.

## Strategy

**Targeted fix.** Host-side adaptive NUM_SPLITS dispatch in `kernel()`. Two compiled
variants (NS=8 and NS=16, both with matching D_CKV_SPLIT for the `s==d` coupling
invariant); pick at dispatch time based on `valid_max`. Pure Python change at the
launcher level; zero kernel edits. Closes the only known mechanistic regression
from exp_51 without risking the established wins.

## Actions (priority ordered)

### 1. Adaptive NUM_SPLITS dispatch on valid_max

**What.** Edit `solution/triton/sparse_fused.py` at the launcher (lines 314-318,
341-375). Keep current kernel bodies unchanged. Add a second compiled launch
path at NUM_SPLITS=8, D_CKV_SPLIT=8, BLOCK_D=64 (the pre-exp_51 tuning, proved
good on 4c46a94b at exp_48). Host-side rule based on `valid_max`:

```python
# Current (exp_51):
D_CKV_SPLIT = 16
BLOCK_D     = D_ckv // D_CKV_SPLIT   # 32
NUM_SPLITS  = 16

# Proposed (exp_54): compute valid_max, dispatch to 8 or 16.
if num_tokens <= 2:
    # Fused path unchanged
    ...
else:
    # valid_max = max over tokens of sum(indices >= 0)
    # Scalar host-side; requires .item() — cost audited below.
    valid_max = int((sparse_indices >= 0).sum(dim=-1).amax().item())
    if valid_max > 1024:
        NUM_SPLITS = 16; D_CKV_SPLIT = 16; BLOCK_D = 32
    else:
        NUM_SPLITS = 8;  D_CKV_SPLIT = 8;  BLOCK_D = 64
    # launch with chosen constexprs
```

**Why.** workload_profile.md derivation (line 75-78 of that file):
- NS=8 + BLOCK_N=128: 1→2 iter boundary at per-token valid=1024.
- NS=16 + BLOCK_N=128: always 1 iter (TOPK=2048 caps it).
- So doubling splits reduces iters only when valid_max > 1024. Below 1024,
  both schemes are 1-iter and NS=16 pays a doubled combine-CTA count +
  barrier-counter target for no iter savings → the +14% regression observed on
  4c46a94b (T=6, valid_max=1002).

The dispatch rule is mechanistic, not statistical: `valid_max > 1024` exactly
characterizes the iter-reduction regime.

**Impact.** Per exp_51/result.md: 4c46a94b regression was 1.6 µs (+14.4%). Full
recovery of that workload at zero cost to the 6 T=7/8 wins (those all have
valid_max ∈ {1091…2048} > 1024, all stay on NS=16). Projected impact on
stride-2: same 16.2 µs of T=7/8 wins + 1.6 µs recovered on 4c46a94b = net
~17.8 µs saved across 7 T≥3 workloads. On the full 23-workload trace set,
possibly more workloads join the NS=8 branch (untracked T=6 workloads).

### 2. Audit `.item()` sync cost, demote to T-based fallback if measured

**What.** Before committing, add a timing probe around the valid_max calculation
inside the launcher on Modal. If the probe shows `>0.5 µs` host-side cost
added to kernel wall-clock (possible via CUPTI measurement of the compute
stream), fall back to a T-based heuristic: route `T == 6` → NS=8, `T ∈ {7, 8}`
→ NS=16. Rationale: in the current benchmark 4c46a94b is the only T=6 in
stride-2; T alone separates it. Less principled but zero overhead.

**Why.** LESSON-20 warns "num_valid not cheaply available from host — .item()
sync costs more than saving." But LESSON-30/41 note CUPTI measures individual
kernel duration, not host stalls. The actual cost in this benchmark harness
needs verification. Cheapest check: A/B comparison of (valid_max dispatch)
vs (T-based dispatch) directly.

**Impact.** De-risks the plan: even if `.item()` IS measured and costly, the
T-heuristic still recovers 4c46a94b.

## Do not try

- **Tree-reduction combine variants** (exp_53, all 3 sub-variants regressed +10-37%).
  LESSON-56: combine is compiler-ILP-max; semantic rewrites lose the online-softmax
  ILP the static_range pattern exposes.
- **`tl.static_range(16)` → `range(16)`** (exp_52 regressed +5-7%). LESSON-55:
  32 KB unroll is the UPPER bound for static_range, not crossover — below that,
  unroll is strictly preferred.
- **BLOCK_N=64 on split path** (exp_49 regressed +22% on T=8). LESSON-52:
  p50-valid=33 does NOT characterize stride-2 dispatched workloads; T=8 tokens
  routinely have ≥512 per-token valid.
- **NUM_SPLITS=4, NUM_SPLITS=32** (exp_31, untested upward). NS=4 closed in exp_31
  at +62%; NS=32 would split combine further without split gain for any workload
  (valid_max ≤ 2048 already fits 1-iter at NS=16).
- **D_CKV_SPLIT_FUSED=1** (exp_36, +17-22% T≤2 regression). Fused path keeps
  D_CKV_SPLIT=8.
- **PDL or cluster launch** (LESSON-27, LESSON-29, LESSON-41). Structurally
  incompatible with atomic-barrier kernel or invisible to CUPTI.
- **`.cg` re-variants, `evict_*` re-variants on already-tested sites.**
  LESSON-47 closes cache-modifier axis; LESSON-48 rules already applied.
- **Gluon migration.** LESSON-46: 6.1× slower on this H=16 shape; axis closed.

## Coordination notes

- **One experiment, two measurement passes.** First A/B tests the valid_max
  dispatch. If win gate passes, log as new best. If it fails due to `.item()`
  cost (T≥3 regresses), IMMEDIATELY pivot to T-based fallback as a follow-up
  (exp_55) using the same mechanism — do not lump them together (one
  optimization per iteration rule).
- **Must compile BOTH kernel variants on first call.** The JIT cache holds both
  variants; subsequent calls reuse them. First call of each T-class pays
  ~1-3 s compile tax (LESSON-37). Modal run will show longer first iteration;
  benchmark uses p50 across iterations so compile is amortized out.
- **Keep the `NUM_SPLITS == D_CKV_SPLIT` invariant (line 347 assert)** — both
  dispatch branches must satisfy it. NS=8 → D_CKV_SPLIT=8, NS=16 → D_CKV_SPLIT=16.
- **Do NOT change the atomic-barrier code path.** `target_count = gen *
  NUM_SPLITS` is already generation-per-key; the `_counter_cache` keys on
  `(device, num_tokens)` not on NUM_SPLITS. Verify: if SAME `num_tokens`
  switches NUM_SPLITS call-to-call, the counter's monotonic invariant might
  break because `target_count` jumps with NUM_SPLITS mid-sequence. Fix: key
  `_counter_cache` and `_generation_cache` on `(device, num_tokens,
  num_splits)` so each dispatch branch has its own counter. CRITICAL
  correctness detail.
- **Success criteria (A/B stride-2 vs exp_51, 2 runs same VM, LESSON-53 noise
  control):**
  - T=6 `4c46a94b` must drop back to ≤ exp_48 latency (~0.011 ms), i.e. same
    direction in both runs and ≥ 10% improvement over exp_51's 0.0125 ms.
  - T=7/8 cluster (6 workloads) must match exp_51 within ±0.0001 ms per
    workload (cannot lose the 2.7 µs/workload NS=16 wins).
  - T≤2 path unchanged → any deltas are noise.
  - Mean Δ ≤ -0.0001 ms across stride-2.
- **Failure criteria (revert):** T=7/8 any workload regresses >2% same direction
  both runs, OR mean Δ >= 0 on T≥3 subset, OR any correctness failure.
- **Measurement protocol.** Quick first (2/2 correctness). Then stride-2 A/B
  `scripts/ab_benchmark.py` with exp_51's snapshotted kernel as baseline, 2
  runs same session. Stratify by T-class per LESSON-51 before judging.

## Cited evidence

- **exp_51/result.md** lines 36-43: "1.6 µs lost per call on 4c46a94b, 10:1
  win/loss ratio" → explicit acknowledgment that this lever is open.
- **workload_profile.md** lines 68-124: Full stride-2 iter-count derivation and
  the dispatch rule `valid_max > 1024 → NS=16 else NS=8`.
- **exp_52/result.md** and **exp_53/result.md**: combine-loop axis closed, push
  decision-maker to structural/dispatch levers.
- **LESSON-54**: "after structural rewrites, re-open tile axes that previously
  regressed" — same principle applied inversely here: NS=8 regressed globally
  before stride-partition, but under modern kernel it's the right choice for
  a specific workload class.
- **LESSON-17**: "Host-level input-characteristic dispatch is free" (regime
  dispatch on T is validated). Extended here to `valid_max`.
- **LESSON-20** (audited): `.item()` sync cost is the single risk — action 2
  de-risks with a T-only fallback.
