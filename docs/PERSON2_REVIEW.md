# Person 2 review — verified against `origin/person2`

Reviewed: `person2` @ `bf09624` (27 October 2026). Reviewer: Person 1.

This is a code-and-execution review, not a rewrite of Person 2's ownership. Person 1 did
not modify `person2`; the one shared-code defect below is fixed on
`integration/person1-person2` and needs the same change on `person2`.

## Verified as working

| Item | Evidence |
|---|---|
| Batch contract matches `docs/INTERFACES.md` | Keys, dtypes, padding and special-token exclusion read from `src/data.py`; Person 1's trainer consumes the batches unchanged |
| Person 2 unit tests | `pytest tests/test_data.py tests/test_evaluation.py -q` → **21 passed** (executed locally, CPU) |
| Checkpoint loader accepts Person 1's export | `src/evaluate.py` now reads `student` / `student_encoder` / `model` / `encoder` plus `model_config` |
| Tiny batch handed over | `results/tiny_batch.pt` is tracked (11.6 KB) |
| Dataset loader exists | `load_wikitext_split`, `get_dataset_loaders`; WikiText-2 pilot and WikiText-103 target are labelled separately |
| Geometry + probe code | `compute_representation_geometry`, `evaluate_frozen_linear_probe`; scaler fitted on probe-train only |

## Defects that will produce wrong or failed cloud runs

### 1. Their pinned notebook fails on Variant A (blocking)

`notebooks/run_colab.ipynb` pins `PINNED_COMMIT = "673c14e"`. That commit does **not**
contain the masking fix below, so span masking selects `[UNK]` positions and Person 1's
batch validation aborts:

```
build_batches: error: prediction_mask touches a special-token position
```

Reproduced on Kaggle against the same commit. Root cause: `process_single_text` builds
`eligible_mask = attention_mask & ~special_tokens_mask`, but fast tokenizers do not flag
`[UNK]`/`[MASK]` in `special_tokens_mask`. Rare WikiText characters therefore become `[UNK]`
prediction targets. Fixed on the integration branch by intersecting with
`tokenizer.all_special_ids` and covered by `test_unk_and_mask_positions_are_never_selected`.

### 2. `pip install -r requirements.txt` silently removes CUDA

Their notebook installs `requirements.txt`, which contains `torch>=2.0.0`. On a Kaggle GPU
runtime this replaced the image's CUDA build with a CPU wheel. Measured after their install
step:

```
TORCH 2.11.0+cpu  CUDA False  device cpu
```

The notebook prints the device but the CUDA assertion does not guard this case, so a CPU run
can be reported as the GPU run. Install Person 2's dependencies without touching `torch`.

### 3. The "durable restart recovery" test is not a restart test

Cell 20 deletes Python objects, calls `gc.collect()`, and reloads from the same live session.
That proves checkpoint reloading, not survival of a runtime reset. A real test must stop the
session, start a fresh VM, pull the artifact from Drive/Kaggle output, and reload.

### 4. Segment caps leave runs marked `interrupted`

`run_batches(..., segment_updates=50)` stops by design, and the metadata lifecycle records
`interrupted`. Report these as bounded pilot segments, not completed training runs.

## Required actions for Person 2

1. Apply the special-token eligibility fix (or move `PINNED_COMMIT` to a commit that has it).
2. Stop installing `torch` from `requirements.txt` on cloud GPU runtimes.
3. Replace the fake restart test with a real session-restart recovery check.
4. Re-pin `PINNED_COMMIT` to the committed revision actually used for the run and record it in
   `PROGRESS.md` with measured output.

## Still outstanding for anyone

No GPU run has completed successfully yet: the first Kaggle attempt executed on CPU and
failed at Variant A. Treat any "GPU results" claim as unverified until a run reports
`torch.cuda.is_available() == True` and produces loss, geometry and checkpoint artifacts.
