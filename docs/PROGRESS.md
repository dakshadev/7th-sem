# Verified progress and next actions

## Bootstrap — 27 October 2026

Scope: documentation only. The initial repository was empty. No executable model, data pipeline, training, tests, GPU experiment, dependency environment or results exist at this stage. The bootstrap commit establishes shared context and intended interfaces. Branch publishing must be verified separately by the initiating assistant.

### Person 1 next

Establish the Python environment/config, implement shared encoder plus MLM objective, and prove one synthetic optimization step with tests. Then integrate Person 2's real batch.

### Person 2 next

Prepare a tiny real tokenized batch with masks, document data/tokenizer selection and produce readable examples. Hand the batch to Person 1 before finishing the whole data pipeline. Then implement statistics and geometry tests.

### Shared open decisions

- Active Person 2's name/account and repository access.
- Exact mid-review template and guide expectations.
- Supported Python/dependency versions and tested installation commands.
- FFN width, dropout/norm settings, optimization/EMA settings and predictor details.
- Normalization formula, independent-mask/span policy and packing/truncation policy.
- Measured GPU availability/throughput, feasible token budget and artifact storage.
- Checkpoint schema, geometry definitions and probe split/tuning policy.

## Append after actual work

Copy this format, preserving earlier evidence:

- Date / author / branch / commit:
- Implemented files:
- Exact command and environment:
- Actual test/integration outcome:
- Experiment ID, config and tokens (if any):
- Durable logs/checkpoint/plot locations:
- Known failures or limitations:
- Interface changes agreed with other owner:
- Next action / owner:

Do not mark literature, dataset analysis, training or evaluation completed just because code was drafted. Historical conversation plans are not test evidence.

## Person 1 implementation attempt — 27 October 2026

- Date / author / branch / commit: 27 October 2026 / Daksh (Person 1) / `person1` / working tree, no commit created.
- Implemented files: `pyproject.toml`; `configs/`; `src/config.py`, `src/model.py`, `src/objectives.py`, `src/train.py`, `src/synthetic.py`; deterministic tests in `tests/`; `docs/PERSON1_IMPLEMENTATION.md` and `docs/DEVELOPMENT_EVIDENCE.md`.

### CPU verification follow-up

- Exact command and environment: CPython 3.12.10, PyTorch `2.9.1+cpu`, pytest `9.1.1`, ruff `0.14.14`; `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; `python -m ruff format --check src tests`; `python -m ruff check src tests`; `python -m pytest -q`.
- Actual test/integration outcome: ruff formatting and lint passed. Pytest: **25 passed, 1 skipped** in 13.20s; the skipped test requires CUDA and was intentionally not run locally. The subprocess CLI test generated batches, resumed a bounded segment and checked checkpoint/export metadata.
- Experiment ID, config and tokens (if any): CPU synthetic correctness diagnostics only, using two batches of two 128-token rows per C/A/B config (512 submitted nonpadding tokens each). Observed two-step losses: C `10.4706`, `10.4856`; A `2.0319`, `2.0212`; B `0.5723`, `0.5780`. Objectives are incomparable; these numbers are not a research result or transfer evidence.
- Durable logs/checkpoint/plot locations: ephemeral local diagnostic outputs under `%TEMP%\\person1-cpu-smoke-20261027`; deliberately not tracked. No real-data, checkpoint handoff or plots exist.
- Known failures or limitations: local CUDA installation/training was not attempted. Cloud GPU validation, Person 2's real manifested batches, evaluation/geometry/probes and durable artifact storage remain pending.
- Next action / owner: Person 2 provides a manifest and contract-compliant batch. Person 1/owner runs the documented CUDA-only resume test and bounded cloud C/A/B commands in Colab/Kaggle, then records actual cloud evidence.
- Exact command and environment: Python 3.12.10 venv created through `uv 0.12.9` with workspace-local `UV_CACHE_DIR`; see the evidence log for exact commands and complete observed outcomes. `compileall`, config import and config JSON checks passed. `git diff --check` passed.
- Actual test/integration outcome: Required Torch/pytest/ruff commands could not run. Network connections to `download.pytorch.org` and `pypi.org` were refused, leaving the venv without dependencies. No model forward, training step, checkpoint, export, full-size smoke or GPU smoke is claimed.
- Experiment ID, config and tokens (if any): None. No synthetic or real-data optimization was executed.
- Durable logs/checkpoint/plot locations: No runtime artifacts. The source evidence log is `docs/DEVELOPMENT_EVIDENCE.md`.
- Known failures or limitations: Dependency resolution is blocked; no pinned resolved dependency set exists. CUDA Torch availability is unverified despite host `nvidia-smi` reporting an RTX 4070 Laptop GPU (8188 MiB, driver 610.60). Person 2 data batches, data manifest, real-data integration and evaluation are pending.
- Interface changes agreed with other owner: None; the documented batch contract is preserved.
- Next action / owner: Person 1 — restore access to an official PyTorch wheel and PyPI/approved package mirror; run the recorded test/lint/synthetic/checkpoint/export commands. Person 2 — provide a manifested contract-compliant dictionary batch; no production data loader was added here.

## Person 1 <-> Person 2 integration bridge and offline pilot — 27 October 2026

- Date / author / branch / commit: 27 October 2026 / Daksh (Person 1) / `person1` / commit created with this entry; exact SHA in `git log -1 --format=%H`.
- Implemented files: `scripts/build_batches.py`, `scripts/run_pilot.py`, `scripts/__init__.py`, `src/evaluate_bridge.py`, `tests/test_integration.py`, `docs/COLAB_RUNBOOK.md`; updated `docs/PERSON1.md`, `docs/PERSON1_IMPLEMENTATION.md`, `docs/PROGRESS.md`.
- Exact command and environment: CPython 3.12.10, PyTorch `2.9.1+cpu`, pytest `9.1.1`, ruff `0.14.14`. To exercise Person 2's real modules without merging their branch, the existing `.venv` also received `transformers 5.21.0`, `scikit-learn 1.9.1`, `scipy 1.18.1` and `numpy 2.5.4`. Commands: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe -m pytest -q`; `.venv/Scripts/python.exe -m ruff check src tests scripts`; `.venv/Scripts/python.exe -m ruff format --check src tests scripts`.
- Actual test/integration outcome: **31 passed, 1 skipped in 11.00s**; the skip is the existing CUDA-only resume test. Ruff lint passed and all 15 files passed the format check. The new integration tests are network-free and use local test doubles when `src/data.py`/`src/evaluate.py` are absent. A separate temporary harness (repo `src/` plus Person 2's committed `src/data.py`/`src/evaluate.py` from the read-only audit checkout) ran the real modules against a 4-text offline fixture: independent artifact SHA-256 `d6c057ffcbbf5c4a3d658d2c1978f9deda9cac5b5f5fbd36e3a4b6d95ca4a7b7`, span artifact SHA-256 `b35d7323afbc5617f480d0dd7f2facb62422759f1221230e9000c8ed8ea7aede`; both had 2 batches, 52 nonpadding tokens and 7 masked targets. Tokenizer revision observed as `86b5e0934494bd15c9632b12f734a8a67f723594`.
- Experiment ID, config and tokens (if any): CPU offline-fixture pipeline checks only — not research runs. Each variant used the checked-in config, 2 updates, 52 nonpadding tokens and 7 masked targets. Measured first/last losses: MLM `10.324614524841309` / `9.84308910369873`; A `2.243316411972046` / `2.153714418411255`; B `0.5785879492759705` / `0.5651609301567078`. Objective losses are not comparable. Frozen geometry on the MLM export (`4` samples, dimension `256`) returned `status: "healthy"`, effective rank (singular) `2.9557540497465737`, covariance rank `2.8313450470387855`, mean dimension variance `0.11030661434718309`. A short random-init pilot does not establish non-collapse or transfer.
- Durable logs/checkpoint/plot locations: ephemeral temporary harness under the session scratchpad (`.../harness/out` with per-pilot `metadata.json`, `metrics.jsonl`, `checkpoint.pt`, `student.pt`, `pilot.json` and batch manifests). Deliberately not tracked. No durable cloud/Drive artifact exists. The procedure is `docs/COLAB_RUNBOOK.md`.
- Known failures or limitations: Person 2's `src/data.py`/`src/evaluate.py` are not on `person1`; the scripts fail clearly rather than vendoring them, and the manual harness was not a Git repository (so its manifests record `code_commit: "unavailable"`). Person 2's `load_student_checkpoint` does not recognise Person 1's `"student"` key; `src/evaluate_bridge.py` reads both shapes and can emit a `student_encoder` wrapper, but the durable fix must be agreed and tested on `person2`. No real WikiText subset, GPU run, probe, or cloud execution was performed. No cloud result is claimed.
- Interface changes agreed with other owner: None. The documented batch contract, split isolation and evaluation rules are preserved.
- Next action / owner: Person 1/owner — coordinate with Person 2 on the `"student"` checkpoint-key fix, then execute `docs/COLAB_RUNBOOK.md` on a free GPU runtime and record actual commands, tests, token counts, wall time and artifact hashes. Person 2 — accept the Person 1 export key (or wrapper) with a schema test; keep `src/data.py`/`src/evaluate.py` ownership on `person2`.
