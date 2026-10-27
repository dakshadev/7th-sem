# Person 2 review — verified against `origin/person2`

Reviewed: `person2` @ `bf09624` (27 October 2026). Reviewer: Person 1.

This is a code-and-execution review, not a rewrite of Person 2's ownership. Person 1 did
not modify `person2`; the one shared-code defect below was fixed on
`integration/person1-person2` and landed on `person2`.

**Status: all four defects below were fixed at `2273c01`, and GPU runs have since completed**
(see `docs/PROGRESS.md`). The findings are kept in their original form so the review still
records what was wrong and why it mattered.

## Verified as working

| Item | Evidence |
|---|---|
| Batch contract matches `docs/INTERFACES.md` | Keys, dtypes, padding and special-token exclusion read from `src/data.py`; Person 1's trainer consumes the batches unchanged |
| Person 2 unit tests | `pytest tests/test_data.py tests/test_evaluation.py -q` → **22 passed** (executed locally, CPU) |
| Checkpoint loader accepts Person 1's export | `src/evaluate.py` now reads `student` / `student_encoder` / `model` / `encoder` plus `model_config` |
| Tiny batch handed over | `results/tiny_batch.pt` is tracked (11.6 KB) |
| Dataset loader exists | `load_wikitext_split`, `get_dataset_loaders`; WikiText-2 pilot and WikiText-103 target are labelled separately |
| Geometry + probe code | `compute_representation_geometry`, `evaluate_frozen_linear_probe`; scaler fitted on probe-train only |

## Defects found (all fixed at `2273c01`)

### 1. Their pinned notebook failed on Variant A (blocking) — **fixed**

`notebooks/run_colab.ipynb` then pinned a commit that predates the masking fix below, so span
masking selected `[UNK]` positions and Person 1's batch validation aborted:

```
build_batches: error: prediction_mask touches a special-token position
```

Reproduced on Kaggle against the same commit. Root cause: `process_single_text` builds
`eligible_mask = attention_mask & ~special_tokens_mask`, but fast tokenizers do not flag
`[UNK]`/`[MASK]` in `special_tokens_mask`. Rare WikiText characters therefore become `[UNK]`
prediction targets. Fixed on the integration branch by intersecting with
`tokenizer.all_special_ids` and covered by `test_unk_and_mask_positions_are_never_selected`;
landed on `person2` at `2273c01`. The notebook now pins `PINNED_COMMIT = "00b8dc8"`.

### 2. `pip install -r requirements.txt` silently removed CUDA — **fixed**

Their notebook installs `requirements.txt`, which contains `torch>=2.0.0`. On a Kaggle GPU
runtime this replaced the image's CUDA build with a CPU wheel. Measured after their install
step:

```
TORCH 2.11.0+cpu  CUDA False  device cpu
```

The notebook prints the device but the CUDA assertion does not guard this case, so a CPU run
can be reported as the GPU run. Install Person 2's dependencies without touching `torch`. The
notebook now installs by name and asserts `torch.cuda.is_available()`; `requirements.txt`
carries a warning against installing torch from it on a GPU runtime.

### 3. The "durable restart recovery" test was not a restart test — **fixed**

Cell 20 deletes Python objects, calls `gc.collect()`, and reloads from the same live session.
That proves checkpoint reloading, not survival of a runtime reset. A real test must stop the
session, start a fresh VM, pull the artifact from Drive/Kaggle output, and reload. The cell is
now labelled a same-session reload, with the real restart requirement documented.

### 4. Segment caps leave runs marked `interrupted` — **fixed**

`run_batches(..., segment_updates=50)` stops by design, and the metadata lifecycle records
`interrupted`. Bounded pilots must be reported as bounded pilot segments, not completed
training runs. The budget-driven runs use `stop_reason: budget` and are documented as such in
`docs/PROGRESS.md`.

## Required actions for Person 2 — all completed at `2273c01`

1. Apply the special-token eligibility fix (or move `PINNED_COMMIT` to a commit that has it).
2. Stop installing `torch` from `requirements.txt` on cloud GPU runtimes.
3. Replace the fake restart test with a real session-restart recovery check.
4. Re-pin `PINNED_COMMIT` to the committed revision actually used for the run and record it in
   `PROGRESS.md` with measured output.

## Status

All four actions above are done on `person2` at `2273c01`, and GPU runs have since completed:
Kaggle T4, torch 2.11.0+cu128, `torch.cuda.is_available() == True`, with three pilot runs and a
12-run budget sweep. Measured numbers are in `docs/PROGRESS.md`; the headline is negative and
is reported as such.
