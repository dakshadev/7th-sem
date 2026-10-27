# Person 1 GPU pilot — Kaggle Tesla T4 (27 October 2026)

**This is a bounded pilot, not the proposed study.** It proves the pipeline runs on a real GPU
and produces comparable artifacts. It does **not** support any claim about which design choice
causes collapse or better transfer.

## Environment (verified inside the run)

| Item | Observed |
|---|---|
| Platform | Kaggle notebook (free tier) |
| GPU | Tesla T4, 15360 MiB |
| Torch | 2.11.0+cu128 |
| `torch.cuda.is_available()` | `True` |
| Device used by trainer | `cuda` |
| Commit | `fbc93db` (`integration/person1-person2`) |

Colab free tier could not be used: it refused to allocate any backend from this location
("Sorry, we were unable to connect to your backend…"). Kaggle entitlement: 30 GPU-hours/week.

## Data

WikiText-2 raw (`Salesforce/wikitext`, `wikitext-2-raw-v1`), train split only, 400 records,
`bert-base-uncased`, length 128, batch 8, mask ratio 0.15, seed 17.

Per artifact: **50 batches, 22,833 non-padding tokens, 3,333 masked targets.**
That is **0.46%** of the proposed ~5M-token budget. Artifact hashes:

- independent masking: `81f3138031f775a0415899b5ba20da3963ab8c8ae68f0f694c8c0752de7f9184`
- span masking: `eec1ee078c5f9477216d43d26177e2ca281196ade9fc21ca7cd663573d8bb0ae`

## Training (50 updates each, identical token budget)

| Variant | First loss | Last loss | Training wall clock | Student SHA-256 |
|---|---|---|---|---|
| C — MLM | 10.5259 | 9.1832 | 2.73 s | `71474288…b8f` |
| A — latent (final-layer target, span masks, attention predictor, MSE) | 2.0140 | 1.3318 | 2.66 s | `52c84fac…515` |
| B — latent (normalized top-2 target, independent masks, linear predictor, Smooth L1) | 0.5635 | 0.4343 | 2.23 s | `f3891586…254` |

Loss scales are objective-specific (cross-entropy vs MSE vs Smooth L1) and **must not** be
ranked against each other. `stop_reason` is `interrupted` for all three: that is the deliberate
50-update segment cap, not a finished training run.

## Frozen representation geometry (WikiText-2 validation, 200 records)

| Student | Effective rank (singular) | Effective rank (covariance) | Status |
|---|---|---|---|
| C — MLM | 70.36 | 2.76 | healthy |
| A — latent | 85.18 | 4.79 | healthy |
| B — latent | 86.88 | 5.45 | healthy |

No variant collapsed at this scale, and none is "better" on this evidence: 50 updates on 23k
tokens cannot separate design choices. **Not run here:** token-level geometry, GPU
throughput/memory benchmark. Both the AG News linear probe and a second seed were run after
this pilot and are recorded in `docs/PERSON1_PROBE.md` and `docs/PERSON1_LONG_PILOT.md`.

## Test suite in the cloud runtime

`python -m pytest -q` → **55 passed, 2 skipped** on the current tree. One skip is the CUDA-only
resume test, the other an integration-only test. The count observed at commit `fbc93db` is
recorded in `docs/PROGRESS.md`; the suite has grown since then.

## Reproduce

```bash
git clone https://github.com/dakshadev/7th-sem.git
cd 7th-sem
pip install "transformers>=4.40" "datasets>=2.18" "scikit-learn>=1.4" "scipy>=1.12" \
  "numpy>=1.26" "matplotlib>=3.8"   # never requirements.txt: it downgrades CUDA torch
python -c "import torch; assert torch.cuda.is_available()"
python -m pytest -q
python scripts/build_batches.py --dataset Salesforce/wikitext --config-name wikitext-2-raw-v1 \
  --split train --tokenizer bert-base-uncased --max-length 128 --policy independent \
  --mask-ratio 0.15 --max-records 400 --batch-size 8 --seed 17 --out artifacts/batches_independent.pt
python scripts/build_batches.py --dataset Salesforce/wikitext --config-name wikitext-2-raw-v1 \
  --split train --tokenizer bert-base-uncased --max-length 128 --policy span \
  --mask-ratio 0.15 --max-records 400 --batch-size 8 --seed 17 --out artifacts/batches_span.pt
python scripts/run_pilot.py --variant mlm --config configs/mlm_c.json \
  --batches artifacts/batches_independent.pt --output artifacts/pilot-mlm --max-updates 50 --device auto
python scripts/run_pilot.py --variant latent_a --config configs/latent_a.json \
  --batches artifacts/batches_span.pt --output artifacts/pilot-latent-a --max-updates 50 --device auto
python scripts/run_pilot.py --variant latent_b --config configs/latent_b.json \
  --batches artifacts/batches_independent.pt --output artifacts/pilot-latent-b --max-updates 50 --device auto
python -m src.evaluate_bridge --student artifacts/pilot-mlm/student.pt \
  --dataset Salesforce/wikitext --config-name wikitext-2-raw-v1 --split validation \
  --tokenizer bert-base-uncased --max-length 128 --max-records 200 --batch-size 8 \
  --device auto --out artifacts/geometry-mlm.json
```

Machine-readable copy of these numbers: `results/person1/kaggle_t4_pilot.json`.

## What this does not show

- No claim that A or B beats MLM, or that normalized targets prevent collapse.
- No transfer evidence (no linear probe was run against a labelled downstream task).
- No replication beyond one seed.
- No measurement of what a full-budget run would cost on free GPUs.
