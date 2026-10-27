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

## GPU pilot — Kaggle Tesla T4 — 27 October 2026

- Date / author / branch / commit: 2026-10-27 / Person 1 / `integration/person1-person2` / `fbc93db`
- Runtime: Kaggle free tier, Tesla T4 15360 MiB, Torch 2.11.0+cu128, `torch.cuda.is_available() == True`, trainer device `cuda`.
- Colab free tier refused to allocate a backend from this location, so Kaggle was used instead.
- Data: WikiText-2 raw train split, 400 records, 50 batches, 22,833 non-padding tokens, 3,333 masked targets (0.46% of the proposed ~5M-token budget).
- Actual test suite in the cloud runtime: `python -m pytest -q` → 53 passed, 1 skipped.
- Measured pilots (50 updates each): MLM 10.5259 → 9.1832 in 2.73 s; latent A 2.0140 → 1.3318 in 2.66 s; latent B 0.5635 → 0.4343 in 2.23 s. All three report `stop_reason: interrupted` because of the deliberate 50-update segment cap.
- Frozen geometry on WikiText-2 validation (200 records): MLM rank(singular/covariance) 70.36 / 2.76; latent A 85.18 / 4.79; latent B 86.88 / 5.45 — all `healthy`, none degenerate.
- Durable evidence: `docs/PERSON1_GPU_PILOT.md`, `results/person1/kaggle_t4_pilot.json`. Student export SHA-256 values are recorded there.
- Known limitations: single seed, bounded subset, no linear probe, no GPU throughput benchmark, objective losses not comparable.
- Interface changes agreed with other owner: none; the span/independent masking fix (special-token eligibility) needs to land on `person2` — see `docs/PERSON2_REVIEW.md`.
- Next action / owner: Person 1 — run the deferred AG News probe and a second seed; Person 2 — apply the masking fix, stop installing `torch` from `requirements.txt`, re-pin the notebook commit.

## Frozen AG News probe — Kaggle T4 — 27 October 2026

- Date / author / branch / commit: 2026-10-27 / Person 1 / `integration/person1-person2` / `c0f330a`
- Runtime: Kaggle Tesla T4, torch 2.11.0+cu128, `cuda.is_available() == True`.
- Probe: AG News, 4000 probe-train / 1000 held-out, scaler+classifier fit on train only, seed 42, `C=1.0`.
- Measured: MLM 0.603 acc / 0.599 macro-F1; latent A 0.598 / 0.596; latent B 0.602 / 0.600; **random-init same-architecture reference 0.632 / 0.628**; majority baseline 0.274.
- Interpretation: every trained student is BELOW the random reference, so 50 updates on 22,833 tokens produced no transfer improvement. The probe validates the harness, not the variants.
- Evidence: `docs/PERSON1_PROBE.md`, `results/person1/probe_agnews_t4.json`.
- Not run: second seed, probe hyperparameter search, token geometry.
- Next action / owner: Person 1 — longer training budget plus a second seed before any between-variant claim.

## Person 2 fixes landed — 27 October 2026

- Branch `person2` fast-forwarded to `2273c01` (also published as `person2-fixes`).
- `src/data.py`: eligibility now intersects `tokenizer.all_special_ids`; regression test fails without the fix and passes with it (22/22 Person 2 tests pass).
- `notebooks/run_colab.ipynb`: installs dependencies by name instead of `requirements.txt` (which had replaced CUDA torch with `2.11.0+cpu`), asserts `torch.cuda.is_available()` with the accelerator-confirmation hint, pins `PINNED_COMMIT = 00b8dc8`, and the "runtime restart" cell is relabelled as a same-session reload with the real restart requirement documented.
- `requirements.txt`: warns that cloud runtimes must not install torch from it.

## Scaled pilot, ablations, probe and Objective 1 — Kaggle T4 — 27 October 2026

- Date / author / branch / commit: 2026-10-27 / Person 1 / `integration/person1-person2` / `ee8676b`
- Runtime: Kaggle Tesla T4 15360 MiB, torch 2.11.0+cu128, `cuda.is_available() == True`. Total GPU time for the batch ~7 minutes.
- Budget per run: 700,482 non-padding tokens, 1,230 updates, 102,721 masked targets (12,000 WikiText-2 records). Stop reason `budget`.
- Anchors, two seeds: MLM 10.526→7.784 (s17) and 10.444→7.945 (s43); latent A 2.014→0.505 / 2.018→0.519; latent B 0.563→0.150 / 0.576→0.154.
- Ablations D–G (seed 17, one change from A): D 0.358, E 0.513, F 0.902, G 0.239 final loss (objective-specific scale; not comparable to A's 0.505).
- Geometry (validation, 300 records, frozen): MLM rank(singular) 10.23/16.95; A 76.43/80.82; B 67.58/72.63; all `healthy`, none degenerate.
- AG News probe (4,000 train / 1,000 held-out): MLM 0.575/0.577, A 0.589/0.588, B 0.593/0.585 accuracy; **random-init reference 0.632**; majority 0.274. Every trained student remains below the random reference.
- Objective 1 ambiguity (400 records, 26,840 positions, single-position masking): mean 1.277 nats, median 0.602, 30.2% of positions above log 5; rare tokens 1.523 nats vs frequent 1.107. Independent 1.612 vs span 1.633 nats, 191/400 records span-more-ambiguous — the earlier 12-record gap does not reproduce.
- Evidence: `docs/PERSON1_LONG_PILOT.md`, `results/person1/{kaggle_long_runs,probe_agnews_long,geometry_long,ambiguity_wikitext2}.json`.
- Known limitations: 14% of the proposed budget; two seeds; D–G single-seed; probe task easy for random features.
- Fixed in passing: `run_pilot` recorded the deprecated `objective.variant` (always empty) instead of `label`; corrected after these runs, so artifact label fields are empty and labels are derived from config paths.
- Next action / owner: Person 1 — full 5M-token budget, then D–G with probes at that budget.

## Budget sweep — 0.69M vs 5M tokens, annealed — 27 October 2026

- Date / author / commit: 2026-10-27 / Person 1 / `5f20e6a`
- Runtime: Kaggle committed kernel `dakshasubramanya/7th-sem-budget-sweep` (v3), GPU T4 x2, torch 2.11.0+cu128, ~50 minutes.
- Design: 3 objectives × 2 seeds × 2 budgets on WikiText-103 (90k records, seq 128, batch 8, mask 0.15). `max_updates` derived from each budget (`budget/575`) so the linear schedule reaches zero in every run.
- Training: 0.69M runs stop at 692,919 tokens / 1,217 updates; 5M runs at 5,000,623 tokens / 8,668 updates; all stop_reason `budget`.
- Probe (AG News, 4,000/1,000, seed 42, frozen): 0.69M → C 0.579/0.574, A 0.591/0.588, B 0.595/0.577. 5M → C 0.529/0.534, A 0.544/0.554, B 0.524/0.550. Random-init 0.632, majority 0.274.
- Findings: (1) no condition beats the random-init encoder at either budget; (2) a 7× larger budget costs every condition 0.04–0.05 accuracy while training loss improves.
- Hypothesis (not yet tested): longer training specialises the encoder to masked input while the probe consumes clean input.
- Evidence: `results/person1/sweep_annealed.json`, `docs/figures/fig_budget.png`.
- Superseded: `results/person1/sweep_pass1_no_annealing.json` — same sweep with `max_updates=40000` for both budgets, so neither schedule annealed. Do not cite as the budget comparison.
- Bugs found and fixed in the kernel: (a) `run_probe.py` launched as a script puts `scripts/` on `sys.path`, so `import src` failed — `PYTHONPATH` now set per subprocess; (b) probe re-run was moved to a separate kernel (`7th-sem-probe-only`) that mounts the sweep output rather than retraining.
