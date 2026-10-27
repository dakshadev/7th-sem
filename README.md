# A Lightweight Self-Supervised Framework for Text Representation Learning Using Masked Latent Prediction

An undergraduate research project (7th semester, MSRUAS) that asks a narrow, testable question:
**when the training signal is a masked latent prediction rather than masked-token prediction, does
a small text encoder learn better or worse representations — and what, specifically, about the
recipe causes that?**

One shared 4-layer Transformer, one data pipeline, one evaluation harness. The only thing that
changes between conditions is the objective. No pretrained weights anywhere in the project
encoder; `bert-base-uncased` is used only as a tokenizer (and, in one analysis, as an external
ambiguity estimator).

## Research question

Masked-token prediction spends its capacity on lexical identity. A masked-latent objective instead
predicts the encoder's *own* hidden states, teacher-side, which promises semantic rather than
lexical targets — but only if the target does not collapse to a constant. This project measures
that trade-off directly, at a scale it can afford, and reports the failure modes (including the
negative ones) rather than only the best cell of the table.

## Conditions

| ID | Signal | Target | Predictor | Loss | Masking |
|---|---|---|---|---|---|
| **C** | token identity (MLM control) | input token | linear head | cross-entropy | independent |
| **A** | masked latent | final teacher layer | attention | MSE | span |
| **B** | masked latent | mean of top-2 normalized teacher layers | linear | Smooth L1 | independent |
| **D–G** | one-at-a-time variants of A | top-2 target (D), span→independent (E), linear predictor (F), Smooth L1 (G) | | | |

The teacher is a frozen EMA copy of the encoder (decay 0.996) whose targets are detached; EMA is
applied after the optimizer step, so the teacher never receives gradients.

## What is built

| Component | State |
|---|---|
| Shared encoder (4 layers, width 256, 4 heads, seq 128), padding-aware attention | Implemented, tested |
| MLM control, latent recipes A/B, EMA teacher, detached targets | Implemented, tested |
| Component ablations D–G as explicit configs (one change from A each) | Implemented, run |
| Data pipeline, independent + span masking, corpus statistics | Implemented (Person 2) |
| Geometry metrics (effective rank), frozen linear probe, ambiguity analysis | Implemented, run |
| Checkpoint save/resume with exact state parity, standalone student export | Implemented, tested |
| Automated tests | **55 passed, 2 skipped** (CUDA-only, plus an integration-only skip) |
| Lint | `ruff check src tests scripts` clean |

## Results so far (all bounded pilots)

Everything below is measured, stored as JSON in `results/person1/`, and regenerated into the
mid-project deck by a script — no number in the slides is hand-typed.

### 1. Scaled pilot — 700,482 tokens per run, two seeds (free Kaggle T4)

12,000 WikiText-2 records, sequence length 128, batch 8, mask ratio 0.15, stop reason `budget`.
Each run is 1,230 updates and 102,721 masked targets, i.e. **14% of the proposed 5M-token budget**.

| Condition | Seed 17 | Seed 43 |
|---|---|---|
| C — MLM | 10.526 → 7.784 | 10.444 → 7.945 |
| A — latent | 2.014 → 0.505 | 2.018 → 0.519 |
| B — latent | 0.563 → 0.150 | 0.576 → 0.154 |

Seed-to-seed spread is small (A: 0.505 vs 0.519). **Losses are objective-specific and are not
comparable across rows** — a lower number does not mean a better model.

### 2. Are the representations collapsed or usable?

Effective rank of the frozen representation on held-out WikiText-2 (300 records):

| Student | Rank (singular) | Rank (covariance) | Status |
|---|---|---|---|
| C — MLM | 10.23 / 16.95 | 1.09 / 1.47 | healthy |
| A — latent | 76.43 / 80.82 | 3.62 / 5.15 | healthy |
| B — latent | 67.58 / 72.63 | 2.87 / 3.57 | healthy |

Nothing collapsed. The MLM control learns a markedly **lower-rank** space than either latent
recipe at this budget. Whether that is caused by the objective or by the MLM head consuming a
larger share of the update budget is not established here.

### 3. The uncomfortable result — frozen linear probe on AG News

4,000 probe-train / 1,000 held-out, identical probe and identical seed for every student.

**Budget sweep, learning rate annealed to zero in every run** (`results/person1/sweep_annealed.json`):

| Budget | C — MLM | A — latent | B — latent | Random-init |
|---|---|---|---|---|
| 0.69M tokens (seed 17 / 43) | 0.579 / 0.574 | 0.591 / 0.588 | 0.595 / 0.577 | **0.632** |
| 5.00M tokens (seed 17 / 43) | 0.529 / 0.534 | 0.544 / 0.554 | 0.524 / 0.550 | **0.632** |

Mean of the two seeds, 0.69M → 5M: C 0.577 → 0.532, A 0.589 → 0.549, B 0.586 → 0.537.
Majority-class baseline: 0.274.

**Two findings, both negative and both consistent across every condition and both seeds:**

1. **No trained student ever beats a randomly initialised encoder.** The best run reaches 0.595
   against 0.632 for the untrained model.
2. **More pretraining makes transfer worse, not better.** A 7× larger budget costs every
   condition 0.04–0.05 accuracy, while the training losses improve over the same range
   (MLM 7.22 → 7.25, latent B 0.146 → 0.088).

The leading explanation is a train/evaluation mismatch: training consumes *masked* input,
the probe consumes *clean* input, and longer training makes the encoder better at the masked
task and less useful as a general text encoder at this scale. That is a hypothesis, not a
finding — the next experiment probes the same checkpoints on masked input to test it.

An earlier pass of this sweep (`results/person1/sweep_pass1_no_annealing.json`) compared both
budgets with `max_updates=40000`, so neither run's learning-rate schedule reached zero. That
confounded budget with schedule; it is kept for the record and must not be cited as the budget
comparison.

### 4. Objective 1 — how ambiguous is masked text, really?

400 records, 26,840 positions, **one position masked at a time** with the surrounding context
intact (masking every position at once destroys the context and yields uniform entropy — an
earlier bug, now fixed). `google-bert/bert-base-uncased` serves only as an ambiguity estimator;
its weights never enter the project encoder.

| Measurement | Value |
|---|---|
| Mean predictive entropy | 1.277 nats |
| Median | 0.602 nats |
| 90th percentile | 3.740 nats |
| Positions above log 5 (≥5 plausible tokens) | 30.2% |
| Positions below 1 nat (near-deterministic) | 60.1% |
| Rare tokens (1–2 occurrences) | 1.523 nats |
| Frequent tokens (> 20) | 1.107 nats |

Rarer tokens are more ambiguous, as expected, and ~30% of positions genuinely have several
plausible fillers — the premise that masked text is not a deterministic target holds.

**But span masking is not more ambiguous than independent masking at this scale:** 1.633 vs 1.612
nats, with 191 of 400 records span-more-ambiguous. An earlier 12-record smoke run suggested a
large gap; with 400 records it disappears. The 400-record result is the one we report.

## Limitations (stated up front)

- Even 5M tokens is three orders of magnitude below BERT-scale pretraining; nothing here speaks
  to what these objectives do at scale.
- Two seeds bound run-to-run stability; they are not confidence intervals.
- Ablations D–G are single-seed, so their differences cannot be separated from seed noise.
- A random 4-layer encoder is a strong baseline on an easy 4-class task.
- The ambiguity estimate comes from a different (pretrained) model than the project encoder.

## Repository layout

```text
configs/                  # one JSON per condition: mlm_c, latent_a, latent_b, ablation_d..g
src/
  model.py                # shared encoder, intermediate hidden states
  objectives.py           # MLM, latent A, latent B, ablation wiring
  train.py                # Trainer, EMA teacher, checkpoints, provenance, run_batches
  config.py               # dataclass configs + runtime validation
  data.py                 # Person 2: tokenization, masking policies, loaders
  evaluate.py             # Person 2: geometry metrics, frozen linear probe
  evaluate_bridge.py      # Person 1 ↔ Person 2 evaluation glue
scripts/
  build_batches.py        # dataset → batch artifact + manifest
  run_pilot.py            # one bounded run with the real trainer
  run_probe.py            # frozen AG News linear probe
  ambiguity_analysis.py   # Objective 1, single-position masking
  build_midreview_deck.py # regenerates docs/mid-project-review.pptx from results/
tests/                    # unit + integration, split by subsystem
results/person1/          # measured evidence, JSON
docs/                     # protocol, interfaces, reviews, runbook, deck
```

## How to reproduce

```bash
git clone https://github.com/dakshadev/7th-sem.git && cd 7th-sem
python -m pytest -q                       # 55 passed, 2 skipped
python scripts/build_batches.py --help    # dataset → batch artifact + manifest
python scripts/run_pilot.py --help        # one bounded pilot
python scripts/run_probe.py --help        # frozen AG News probe
python scripts/ambiguity_analysis.py --help
python scripts/build_midreview_deck.py    # rebuild the deck from results/
```

Compute: **free GPU only** — a Kaggle T4 is sufficient (30 GPU-hours/week). Colab's free tier
refused to allocate a backend from our location, which is why the runbook targets Kaggle. On a GPU
runtime do **not** `pip install -r requirements.txt`: it pins a CPU wheel over the CUDA build.

Full run instructions: [docs/COLAB_RUNBOOK.md](docs/COLAB_RUNBOOK.md). Every run records its
commit, device, token budget, wall-clock time and artifact hashes; runs that were interrupted are
labelled as such rather than reported as finished.

## Ownership and branches

| Branch | Owner | Responsibility |
|---|---|---|
| `main` | both, integration only | canonical integrated tree — every result in `results/` came from a commit on it |
| `person1` | Daksh | encoder, objectives, training, configs, model tests |
| `person2` | teammate | data, masking, evaluation, plots, data/evaluation tests |

One shared encoder and one runner; never two incompatible models. Changes to shared code go to
`main` and are reported to the other owner, who replicates them on their branch.

## Documentation map

- [AGENTS.md](AGENTS.md) — instructions for any AI assistant working in this repo.
- [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) — proposal, scope, deadlines.
- [docs/INTERFACES.md](docs/INTERFACES.md) — the batch/evaluation contract both owners code against.
- [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) — what counts as a valid run.
- [docs/PROGRESS.md](docs/PROGRESS.md) — chronological, measured evidence log.
- [docs/PERSON1_LONG_PILOT.md](docs/PERSON1_LONG_PILOT.md) — scaled pilot, ablations, probe, ambiguity.
- [docs/PERSON2_REVIEW.md](docs/PERSON2_REVIEW.md) — review of the data/evaluation modules.
- [docs/COLAB_RUNBOOK.md](docs/COLAB_RUNBOOK.md) — GPU runbook (Kaggle).
- [docs/mid-project-review.pptx](docs/mid-project-review.pptx) — mid-project deck, generated from results.

## Roadmap to the final submission

1. ~~**Budget sweep** — 0.7M → 5M tokens on WikiText-103, C/A/B × 2 seeds, probe at each tier.~~
   **Done.** Answer: transfer does not overtake the reference at either budget; it degrades.
   Next experiment isolates the masked-versus-clean input mismatch as the cause.
2. **Ablations at full budget** with probes, two seeds — currently loss-only, single-seed.
3. **SIGReg (recipe H)** — proposed but not yet implemented.
4. **Second probe (SST-2)** so that one easy task cannot carry the conclusion.
5. Report, final deck, viva preparation.

## Collaboration rules

Work on your own branch, commit focused changes, open a PR to `main`; the other owner reviews
interfaces and tests. Do not force-push shared branches. Keep datasets, checkpoints, virtual
environments and credentials out of Git. Record durable artifacts (and their hashes) outside
ephemeral GPU runtimes — a Kaggle or Colab runtime is wiped when the session ends.

The repository is public, so Colab and Kaggle can clone it without a token. Never commit a token.
