# Data integrity notes

An independent audit re-derived every reported number from the raw JSON in
`results/person1/` and from the Kaggle run trees, then checked them against each other.
This file records what held, and the six places where the data can be misread.

Purpose: anyone citing these results (including us, in the final report) should know which
field is authoritative and how much of a difference is signal.

## Verified

- **Grid complete:** 12 runs, exactly one per (objective × seed × budget) cell.
- **Provenance:** all 12 carry `git_commit 5f20e6a29c` with `dirty=false`, and 12 distinct
  `student.pt` SHA-256 values.
- **Probe attachment:** every `probe-sweep` entry's `student_sha256` matches the on-disk
  `student.pt` for that run, so the accuracy numbers are tied to the weights being reported.
- **Reconciliation:** `pilot.measured.updates` equals the `metrics.jsonl` line count,
  `nonpadding_tokens` equals the sum of the log's per-step tokens, and
  `first/last_train_loss` equal the first and last logged losses — in all 12 runs.
- **Stopping:** `stop_reason == "budget"` in all 12, with token counts within one batch of the
  target (700,482 vs 700,000; 5,000,623 vs 5,000,000).
- **Transfer claim:** every student is below the random baseline, margins 0.039–0.107.
- **Geometry coverage:** 12 `geom-*.json`, one per run, all `status: measured`,
  `is_degenerate: false`, same 300-sample evaluation set.

## Six things that look like errors and are not

### 1. WikiText-2 and WikiText-103 produce identical 700k runs

The 0.69M runs are bit-identical whether the run is labelled `wikitext-2-raw-v1` or
`wikitext-103-raw-v1`, including `first_train_loss`. That is expected: **the WikiText-2 train
split is the prefix of the WikiText-103 train split.** Verified directly — the first five
records of both configs are byte-identical (same `= Valkyria Chronicles III =` article).

A 700k-token budget only consumes the overlapping prefix, so the two corpora are the same
data at that budget. The 5M runs are the ones that draw on the larger corpus. Do not describe
the 0.69M tier as "wikitext-2 vs wikitext-103" anything; at that budget it is one corpus.

### 2. `pilot.json` records no seed

There is no `seed` key in any `pilot.json` or `geom-*.json`. The seed exists only in
`metadata.json: config.run.seed` and in the run directory name. `results/person1/sweep_annealed.json`
derives its `seed` field from the directory name for that reason. First-batch losses do agree
bit-for-bit between the two budgets at the same seed, which is consistent with a shared seed
stream — but the file to quote for the seed is `metadata.json`.

### 3. Two different "training seconds" per run

`pilot.json: measured.training_wall_clock_seconds` is 7–8% larger than
`metadata.json: counters.training_seconds` in every run (e.g. 337.08 vs 312.86 for
`b5000000-s17-mlm`). The pilot field includes the batch-fetch and checkpoint overhead; the
metadata field is the sum of per-step training time from `metrics.jsonl`. Use the pilot field
when quoting wall clock, the metadata field when quoting optimisation time, and never mix them.

### 4. `data.counts.nonpadding_tokens` is the stream size, not the budget

`pilot.json: data.counts.nonpadding_tokens` is `6480948` in all 12 runs — the size of the
batch artifact, not the tokens consumed. The consumed figure is
`pilot.measured.nonpadding_tokens` (700,482 / 5,000,623). The field name invites exactly the
wrong reading.

### 5. The random-initialised reference is not hash-pinned

`probe-*.json: random_baseline.student_sha256` is `null`: the reference is a fresh random
initialisation inside each probe run, so it is not reproducible from a saved artifact. It
therefore drifts a little between runs — 0.632 in one probe file, 0.631 in another. Both are
above every trained student, which is the claim we make; quote it as ≈0.63, not 0.632.

### 6. Identical weights, slightly different probe accuracy

Re-probing hash-identical weights gives slightly different numbers (0.593 vs 0.592 accuracy;
macro-F1 differing in the fourth decimal), because the probe fits a `StandardScaler` +
`LogisticRegression` whose solver and seed are fixed per probe run, not per student.

**Consequence: treat any difference below ~0.002 accuracy as noise.** The budget effect we
report is 0.04–0.05 — twenty to thirty times that floor, which is why it is safe to call it
real. Chasing individual seed-level gaps smaller than 0.002 is not.

## Never pool across experiment passes

Two passes of the same nominal configuration exist (the superseded non-annealed sweep and the
annealed one). Runs with the same label, seed and budget in different passes are **different
trainings with different weights and different hashes**. Compare within a pass only; the
superseded pass is marked as such in its own file.

## Reproducing the audit

The checks above are mechanical, not testimonials:

```bash
python -c "
import json, glob
for p in sorted(glob.glob('results/person1/*.json')):
    d = json.load(open(p))
    print(p, d.get('schema_version'), list(d)[:6])
"
```

Every number in this file came from those files or from the Kaggle run tree; nothing here was
estimated.
