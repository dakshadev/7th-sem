# Person 1 implementation

## Status and environment

This branch contains the Person 1 implementation and deterministic test suite for the encoder, C/A/B objectives, fp32 runner, checkpointing, student export, the batch/pilot orchestration scripts, and the evaluation bridge. It is runtime-verified in the CPU environment: PyTorch `2.9.1+cpu`, pytest `9.1.1` and ruff `0.14.14`; **55 tests passed, 2 skipped** (one CUDA-only resume test, one integration-only test). The same suite has run on Kaggle T4 in a CUDA build. `docs/PROGRESS.md` records the commands and output.

The selected interpreter is `C:\Users\daksh\AppData\Local\Programs\Python\Python312\python.exe`; `.venv` was created from it using `uv 0.12.9`. The project intentionally has no pinned lockfile yet: dependency resolution could not complete locally, and inventing a resolved version would be misleading. CUDA verified in the cloud runtime (torch 2.11.0+cu128); the local host is CPU-only by policy.

The tested local CPU commands are:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv --python "C:\Users\daksh\AppData\Local\Programs\Python\Python312\python.exe" .venv
uv pip install --python .venv\Scripts\python.exe torch==2.9.1 pytest==9.1.1 ruff==0.14.14
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

For a Colab/Kaggle GPU runtime, install a CUDA-enabled PyTorch build compatible with that runtime, install this project without replacing that build (`pip install -e . --no-deps`), then run `pytest -q` and the bounded C/A/B commands below. Do not report a GPU result until those commands actually complete. The exported loader returns an evaluation-mode encoder by default; downstream code owns any explicit `requires_grad_(False)` freeze.

## Interfaces

`src.model.SmallEncoder(input_ids, attention_mask, return_hidden_states=False)` implements a random-initialized learned token/absolute-position Transformer. Its defaults are vocabulary 30,522, length 128, width 256, four layers, four heads, FFN 1024, pre-norm and dropout 0.1. `attention_mask=True` means a valid token; it is inverted at PyTorch's `src_key_padding_mask` boundary. Every post-layer state is available when requested; embeddings are never included. Empty rows, non-int64 IDs, out-of-range IDs and lengths outside the configured range fail explicitly. Padded vectors are reset to zero after each layer, so padded IDs cannot alter valid outputs in eval mode.

The runner accepts Person 2's documented CPU tensor dictionary exactly: `input_ids`, `masked_input_ids`, `attention_mask`, `prediction_mask`. It rejects a selected padding or configured special-token location. It owns no text loading, tokenizer, splitting, masking sampler or corpus packing. `src.synthetic` is a diagnostic-only fixture producer.

## Objectives and information flow

Student always sees `masked_input_ids`; latent teacher always sees uncorrupted `input_ids` under `no_grad()` and `eval()`. The teacher is copied from the student at construction, has no trainable gradients, and receives EMA only after a finite successful optimizer step:

`teacher <- ema_decay * teacher + (1 - ema_decay) * student`.

No-target batch returns a safe skip and does not advance update/token/masked-target counters or EMA. The runner is deliberately fp32 only; AMP/scaler is rejected in config rather than ambiguously pretending skipped scaler updates are handled.

| Variant | target / loss | predictor semantics |
|---|---|---|
| C `mlm` | Original IDs at selected positions; cross entropy | Vocabulary head on full student contextual states. |
| A `latent_a` | Detached final teacher layer at selected span-mask positions; mean squared error | Exactly two Transformer attention layers run on the complete contextual masked student sequence. They receive the valid-token mask and add learned absolute predictor positions; they never receive teacher/unmasked embeddings. |
| B `latent_b` | Detached average of individually normalized final two **post-layer** teacher states; Smooth L1 (`beta=1`) | Linear vector projection of student contextual masked states. |

For B normalization, every width vector is layer-normalized over its last/feature axis, with `unbiased=False` variance and epsilon `1e-5`: `(x - mean_last) / sqrt(var_last + 1e-5)`. The two normalized layers are then averaged. Losses reduce PyTorch's default mean over selected positions and feature dimensions only.

## Configuration, logging, and checkpoint contract

`src.config` is a stdlib dataclass validator (Pydantic is not required): it explicitly covers model FFN/dropout/norm, AdamW LR/weight decay/betas/gradient clipping, linear warmup/decay update schedule, EMA, recipe mask metadata (independent or fixed-length 3-token spans at ratio 0.15), predictor/dropout/loss beta, seed and token budget. Person 2 remains responsible for implementing and documenting the actual deterministic mask sampler/boundary policy. The checked-in `configs/` JSON files declare C/A/B defaults. Token budget counts all nonpadding tokens of the submitted batch. The final batch is processed whole if it crosses the budget; its overshoot is recorded rather than silently sliced.

`run_batches` streams and flushes `metrics.jsonl` after each completed batch, preserving earlier segments on resume. It records atomic lifecycle metadata, source hashes, actual trainer device, timing/counters, batch-artifact hash and optional external manifest. It checks update/token budgets before fetching the next batch; a full final batch may overshoot only by that batch. The external producer must supply real dataset/tokenizer/split manifests; absence is explicit.

Checkpoint schema 1 stores student, objective head/predictor, teacher, AdamW optimizer, scheduler, counters, Python/Torch CPU (and CUDA when available) PRNG state, config and optional data state. Atomic replacement is used. Checkpoints always load on CPU first, so CPU RNG byte tensors never move to CUDA; optimizer state then follows loaded parameters. Runtime output path/device/checkpoint cadence may differ on resume, but architecture and training semantics cannot. `StatefulBatchSequence` persists only the last successfully processed position; generic `DataLoader` users must supply equivalent sampler/data state or accept that only model/optimizer/RNG state resumes exactly. `export_student` writes only model config and student state; `load_exported_student` lets Person 2 load it without a teacher or objective.

## Integration bridge, pilot orchestration and runbook

`scripts/build_batches.py` turns one fixed text split into the documented batch artifact
(`torch.save(list[dict], path)`) by calling Person 2's `src/data.py` lazily; it never vendors
or reimplements their tokenizer/masking code. It requires exactly one split, rejects joined
split names, validates every emitted batch (keys, dtypes, shapes, special-token and padding
exclusion, nonempty targets) and writes a sibling JSON manifest with the dataset/config/
split request, tokenizer ID and observed cache revision, counts, policy/ratio/seed, Git
commit, artifact SHA-256 and explicit bounded-subset limitations. `--offline-fixture` reads
a local JSON text list for network-free checks; raw datasets are never written to the repo.

`scripts/run_pilot.py` loads a checked-in config, reuses or builds a manifested artifact,
constructs the real `Trainer`, calls `run_batches` and exports `student.pt`. It writes
`metadata.json`, `metrics.jsonl`, `checkpoint.pt`, `student.pt` and a measured `pilot.json`
(wall-clock, step time, tokens, masked targets, first/last loss, stop reason). `--max-updates`
is a segment cap and does not alter the configured learning-rate schedule; `--device auto`
selects CUDA only when available. No metric is synthesised: unmeasured fields are `null` or
listed under `not_run`.

`src/evaluate_bridge.py` loads either Person 1's standalone export or a nested training
checkpoint, freezes the student in evaluation mode, embeds uncorrupted held-out sequences
under `torch.no_grad()`, and calls Person 2's `masked_mean_pooling` and
`compute_representation_geometry`. Export paths use `src.train.load_exported_student`.
Unavailable metrics (token geometry, GPU throughput benchmark, optional pretrained-BERT
analysis) are reported as `not_run`. The cross-branch checkpoint defect is resolved:
`src/evaluate.py` accepts `student`, `student_encoder`, `model` and `encoder`.

Verification then went past the CPU pipeline check. The measured runs are in `results/person1/`
and `docs/PERSON1_LONG_PILOT.md`: a 700,482-token pilot on 12,000 WikiText-2 records with two
seeds, the D-G ablations, frozen geometry, the AG News probe, the Objective 1 ambiguity
analysis and a 12-run budget sweep. The headline result is negative, and it is reported as
such.

## Commands after dependencies resolve

```powershell
# Diagnostic-only tiny C/A/B and checkpoint/export artifact
.\.venv\Scripts\python.exe -m src.synthetic --output results\synthetic.pt --count 4 --batch-size 2 --length 128
.\.venv\Scripts\python.exe -m src.train --config configs\mlm_c.json --batches results\synthetic.pt --output results\mlm-smoke
.\.venv\Scripts\python.exe -m src.train --config configs\latent_a.json --batches results\synthetic.pt --output results\latent-a-smoke
.\.venv\Scripts\python.exe -m src.train --config configs\latent_b.json --batches results\synthetic.pt --output results\latent-b-smoke
```

Those are implementation diagnostics only. The real-data and GPU pilots that followed are
recorded in `results/person1/` and `docs/PERSON1_LONG_PILOT.md`; see `docs/COLAB_RUNBOOK.md`
for the procedure that produced them. The AG News probe result is a measured negative, not a
claim that any latent condition transfers better than MLM.
