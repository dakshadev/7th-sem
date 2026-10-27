# Scaled pilot and Objective 1 — free Kaggle T4 (27 October 2026)

Runtime verified inside the run: `Tesla T4, 15360 MiB`, `torch 2.11.0+cu128`,
`cuda.is_available() == True`, trainer device `cuda`. Commit `ee8676b`.
Total GPU time for this batch: **~7 minutes**. No paid compute, no local CUDA.

## Budget

| Item | Value |
|---|---|
| Dataset | `Salesforce/wikitext`, `wikitext-2-raw-v1`, train split, 12,000 records |
| Per-run tokens | **700,482** non-padding (14% of the proposed 5M-token budget) |
| Per-run updates | 1,230 |
| Per-run masked targets | 102,721 |
| Stop reason | `budget` (token budget reached, not a step cap) |

## Anchors — two seeds

| Seed | Condition | First loss | Last loss | Train time |
|---|---|---|---|---|
| 17 | C — MLM | 10.5259 | 7.7839 | 47.2 s |
| 43 | C — MLM | 10.4439 | 7.9446 | 49.4 s |
| 17 | A — latent (span, final layer, attention, MSE) | 2.0140 | 0.5052 | 44.6 s |
| 43 | A — latent | 2.0181 | 0.5187 | 45.0 s |
| 17 | B — latent (independent, top-2 normalized, linear, Smooth L1) | 0.5635 | 0.1500 | 36.0 s |
| 43 | B — latent | 0.5758 | 0.1544 | 35.2 s |

Seed-to-seed spread is small (A: 0.505 vs 0.519; B: 0.150 vs 0.154; C: 7.78 vs 7.94), so the
training path is stable. **Losses remain objective-specific and are not comparable across rows.**

## Component ablations D–G — one change from A, seed 17

| Run | Change vs A | First loss | Last loss | Train time |
|---|---|---|---|---|
| D | teacher target → normalized top-2 | 1.8182 | 0.3584 | 44.9 s |
| E | masking → independent tokens | 2.0371 | 0.5132 | 45.3 s |
| F | predictor → linear | 1.6133 | 0.9015 | 35.3 s |
| G | loss → Smooth L1 | 0.7221 | 0.2395 | 44.8 s |
| A (reference) | — | 2.0140 | 0.5052 | 44.6 s |

Because the objectives differ, the only readable signal here is *within* A and each ablation's
own curve. G's lower values reflect Smooth L1's smaller scale, not a better model.

## Frozen representation geometry — held-out validation, 300 records

| Student | Rank (singular) | Rank (covariance) | Status |
|---|---|---|---|
| C MLM (seed 17) | 10.23 | 1.09 | healthy |
| C MLM (seed 43) | 16.95 | 1.47 | healthy |
| A latent (17) | 76.43 | 3.62 | healthy |
| A latent (43) | 80.82 | 5.15 | healthy |
| B latent (17) | 67.58 | 2.87 | healthy |
| B latent (43) | 72.63 | 3.57 | healthy |

Two observations worth stating: nothing collapsed, and the **MLM control learns a markedly
lower-rank representation** than either latent recipe at this budget (rank ≈ 10–17 versus 68–81).
Whether that reflects the objective or the fact that MLM's head consumes a much larger fraction
of the update budget is not established here.

## AG News frozen linear probe — 4,000 probe-train / 1,000 held-out

| Student | Accuracy | Macro-F1 |
|---|---|---|
| C MLM (17) | 0.575 | 0.570 |
| C MLM (43) | 0.577 | 0.568 |
| A latent (17) | 0.589 | 0.586 |
| A latent (43) | 0.588 | 0.582 |
| B latent (17) | 0.593 | 0.590 |
| B latent (43) | 0.585 | 0.579 |
| **Random-initialised reference (same architecture)** | **0.632** | **0.628** |
| Majority baseline | 0.274 | — |

**Every trained student still scores below the random reference.** At 700k tokens the encoders
have not produced features that beat an untrained encoder on this probe. This is a real,
measured, negative result and it is the most important finding of the milestone: at three
orders of magnitude below BERT-scale pretraining (3.3B tokens), the bottleneck is budget, not
recipe. One contributing factor is the train/evaluation mismatch inherent in the contract — the
student is trained on masked inputs and evaluated on clean ones — which a long enough run
overcomes but a 0.7M-token run does not.

## Objective 1 — masked-word ambiguity

400 records, 26,840 eligible positions, single-position masking with the surrounding context
intact. Pretrained `google-bert/bert-base-uncased` is used **only** as an ambiguity estimator;
its weights never touch the project encoder.

| Measurement | Value |
|---|---|
| Mean predictive entropy | 1.277 nats |
| Median predictive entropy | 0.602 nats |
| 90th percentile | 3.740 nats |
| Positions above log 5 (≈5 plausible tokens) | 30.2 % |
| Positions below 1 nat (near-deterministic) | 60.1 % |
| Rare tokens (1–2 occurrences) | 1.523 nats |
| Mid-frequency (3–20) | 1.433 nats |
| Frequent (> 20) | 1.107 nats |

The frequency gradient is the expected one: rarer tokens are more ambiguous. About 30 % of
positions genuinely have several plausible fillers, which supports the proposal's premise that
masked text is not a deterministic target.

**But the span-versus-independent comparison does not confirm the expected difference at this
scale:** independent-mask entropy 1.612 vs span 1.633 nats, with 191 records where spans are more
ambiguous and 202 where independent masking is. An earlier 12-record smoke run showed a large
gap (7 of 8 records); with 400 records it vanishes. We report the 400-record result.

## Limitations

- 700k tokens is 14% of the proposed budget and far below the regime where the literature's
  claims were established.
- Two seeds bound stability, not confidence intervals.
- D–G were run once each, so their loss differences cannot be separated from seed noise.
- The probe's random reference is strong because AG News with 4,000 labels is easy.
- Pretrained-BERT entropy is an external estimate of ambiguity, not the project encoder's.

## Reproduce

See `docs/COLAB_RUNBOOK.md`. Machine-readable evidence:
`results/person1/kaggle_long_runs.json`, `results/person1/probe_agnews_long.json`,
`results/person1/geometry_long.json`, `results/person1/ambiguity_wikitext2.json`.
