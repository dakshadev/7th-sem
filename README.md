# Lightweight Self-Supervised Text Representation Learning

**Project:** A Lightweight Self-Supervised Framework for Text Representation Learning Using Masked Latent Prediction.

This undergraduate research project investigates how target construction, masking, predictor design and regression loss affect collapse and transfer in a small text encoder. It compares masked latent prediction against a masked-language-modeling (MLM) control.

## Current status — 27 October 2026

The experimental system is built and has been run on a free GPU. What is missing is training
budget, not infrastructure.

| Area | State |
|---|---|
| Encoder, MLM control, latent recipes A/B, EMA teacher | Implemented and tested |
| Component ablations D–G (one change at a time) | Configs + runner ready |
| Data pipeline, masking policies, corpus statistics | Implemented (Person 2) |
| Geometry metrics, frozen linear probe, ambiguity analysis | Implemented and run |
| Checkpoint resume, standalone student export | Tested, including exact state parity |
| Automated tests | 55 passed, 2 skipped (CUDA-only + integrated-tree skip) |
| Lint | `ruff check src tests scripts` clean |
| GPU runs | Kaggle free T4 (Colab free tier refused a backend from our location) |

**Everything reported so far is a bounded pilot, not the proposed ~5M-token study.** The first
50-update pilot produced representations worse than a random-init encoder on AG News; that
negative result is recorded rather than hidden, and it is why the next step is budget.

Start with [GPU pilot](docs/PERSON1_GPU_PILOT.md), [probe](docs/PERSON1_PROBE.md),
[Person 2 review](docs/PERSON2_REVIEW.md) and the [progress log](docs/PROGRESS.md).

## Start here

1. Read [AGENTS.md](AGENTS.md) — instructions for any AI assistant.
2. Read [project context](docs/PROJECT_CONTEXT.md) — proposal, scope and deadlines.
3. Agree on [shared interfaces](docs/INTERFACES.md) before implementation.
4. Follow your handoff: [Person 1](docs/PERSON1.md) / [Person 2](docs/PERSON2.md).
5. Use [experiment protocol](docs/EXPERIMENT_PROTOCOL.md) and update [progress](docs/PROGRESS.md) after verified work.

## Ownership

| Branch | Owner | Responsibilities |
|---|---|---|
| `main` | Both, integration only | Reviewed shared foundation and integrated work |
| `person1` | Daksh | Encoder, MLM/latent objectives, training, EMA, checkpoints, model tests |
| `person2` | Teammate (name unconfirmed) | Data, masking, statistics, evaluation, plots, Colab launcher, data/evaluation tests |

`main` is the canonical integrated branch: both owners' work lives there together, and every
result in `results/` was produced from a commit on it. `person1` and `person2` remain the owners'
working branches. One shared encoder and runner; never implement separate incompatible models.

## How to reproduce

```bash
git clone https://github.com/dakshadev/7th-sem.git
cd 7th-sem
python -m pytest -q                     # 55 passed, 2 skipped
python scripts/build_batches.py --help  # dataset → batch artifact + manifest
python scripts/run_pilot.py --help      # one bounded pilot with the real trainer
python scripts/run_probe.py --help      # frozen AG News linear probe
python scripts/ambiguity_analysis.py --help   # Objective 1 ambiguity analysis
```

The GPU path is documented end to end in [docs/COLAB_RUNBOOK.md](docs/COLAB_RUNBOOK.md): a free
Kaggle T4 is sufficient; do not install `torch` from `requirements.txt` on a GPU runtime.

## Local checkout

Daksh's checkout: `C:\Users\daksh\MyLife\Work\7th-sem`.

```bash
git clone https://github.com/dakshadev/7th-sem.git
cd 7th-sem
git switch person1   # teammate uses person2
```

The repository is public, so Colab and Kaggle can clone it without a token. Never put tokens in notebooks or commits.

## Layout

```text
configs/                 # Small explicit experiment configurations
src/
  data.py                # Person 2: tokenization, splits, masking, loaders
  model.py               # Person 1: shared encoder
  objectives.py          # Person 1: MLM and masked latent objectives
  train.py               # Person 1: runner, EMA, checkpoints, logs
  evaluate.py            # Person 2: representation metrics and frozen probes
notebooks/run_colab.ipynb # Person 2, jointly verified
tests/                   # Split ownership by subsystem
```

The supported target interpreter is Python 3.12. The verified local CPU environment is PyTorch 2.9.1+cpu, pytest 9.1.1 and ruff 0.14.14; cloud GPU commands remain explicitly unexecuted.

## Collaboration

Work on your assigned branch. Commit focused changes and push your branch, then open a pull request to `main`. The other member reviews tests and interfaces. Merge small compatible increments, then sync `main` into your branch:

```bash
git fetch origin
git merge origin/main
# resolve conflicts with the other owner; do not overwrite their work
```

Do not force-push shared branches. Keep datasets, checkpoints, virtual environments, credentials and large artifacts out of Git. Store durable artifacts outside ephemeral GPU runtimes and record their locations/checksums. Exact run evidence belongs in the progress log and experiment manifest.
