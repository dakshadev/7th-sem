# Person 1 — frozen AG News linear probe (Kaggle T4, 27 October 2026)

Bounded probe of the three pilot students plus a random-initialised reference of the same
architecture. Runtime verified inside the run: `Tesla T4`, `torch 2.11.0+cu128`,
`cuda.is_available() == True`; commit `c0f330a`.

## Setup

- Probe task: AG News (`fancyzhx/ag_news`), 4 classes.
- Probe-train 4000 examples, held-out test 1000 examples; majority baseline 0.274.
- Student frozen in eval mode, **uncorrupted** text, Person 2's attention-mask-aware mean pooling
  that excludes padding and every special token.
- `StandardScaler` + `LogisticRegression(C=1.0, max_iter=1000, lbfgs)` fit on probe-train only;
  the test split is used once, for scoring. Fixed seed 42, no probe hyperparameter search.

## Measured results

| Model | Test accuracy | Macro-F1 | Majority baseline |
|---|---|---|---|
| C — MLM (50 updates) | 0.603 | 0.599 | 0.274 |
| A — latent, span + final-layer target + attention predictor + MSE | 0.598 | 0.596 | 0.274 |
| B — latent, independent + normalized top-2 + linear + Smooth L1 | 0.602 | 0.600 | 0.274 |
| **Random-initialised encoder (same architecture)** | **0.632** | **0.628** | 0.274 |

## What this actually means

**Every trained student scores below the random-initialised reference.** With 50 updates over
22,833 tokens the encoders have not learned representations that transfer better than an
untrained one. Two consequences:

1. The probe harness works end to end: features, pooling, frozen evaluation and the
   train-only-fit invariance all execute and produce plausible numbers.
2. No comparison between C, A and B is meaningful at this budget. The differences among them
   (0.598–0.603) are smaller than the gap to the random baseline (0.632).

The random reference scoring 0.63 with 4000 labelled examples also shows AG News is easy enough
that a weak encoder passes it; a probe on this budget cannot separate design choices. Any claim
about which component matters requires substantially longer training and a second seed.

## Not run

- Second seed (no replication).
- Probe hyperparameter search (fixed `C=1.0` by design, to avoid tuning on the evaluation set).
- Token-level geometry.
- Larger probe splits.

Machine-readable record: `results/person1/probe_agnews_t4.json`.
