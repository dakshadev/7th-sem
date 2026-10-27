# Lightweight Self-Supervised Text Representation Learning

**Project:** A Lightweight Self-Supervised Framework for Text Representation Learning Using Masked Latent Prediction.

This undergraduate research project investigates how target construction, masking, predictor design and regression loss affect collapse and transfer in a small text encoder. It compares masked latent prediction against a masked-language-modeling (MLM) control.

## Current status — 27 October 2026

Documentation and collaboration bootstrap only. No model, dataset pipeline, training runs, checkpoints, benchmarks, dependency environment or executable notebook has been implemented or verified in this repository. Do not present this setup as experimental progress. Mid-project presentation: **29 October 2026**. Compute: free Colab/Kaggle GPU, availability unverified.

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

One shared encoder and runner; do not implement separate incompatible models on each branch. Both members interpret results and prepare the presentation. Do not merge another person's unfinished work without agreement.

## Immediate milestone

Run a single real-data MLM optimization step end to end. Before integration, Person 1 can test with synthetic batches while Person 2 tests data and metrics with fixtures. Synthetic tests are not research results.

**Review minimum:** working MLM and one latent recipe, saved checkpoints, real loss and representation diagnostic plots. **Preferred:** matched-budget A/B/C pilots and an initial AG News frozen probe. No guarantee these fit the available time/GPU quota. Defer full ablations, repeated seeds, SIGReg and SST-2 until the core works.

## Local checkout

Daksh's checkout: `C:\Users\daksh\MyLife\Work\7th-sem`.

```bash
git clone https://github.com/dakshadev/7th-sem.git
cd 7th-sem
git switch person1   # teammate uses person2
```

The repository is private; each collaborator needs access. Never put tokens in notebooks or commits.

## Proposed layout (not implemented yet)

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

Use PyTorch and a BERT tokenizer; no dependency versions are selected yet. Implementation must establish a supported Python environment, explicit dependencies and tested commands. There is deliberately no fictional install/training command here.

## Collaboration

Work on your assigned branch. Commit focused changes and push your branch, then open a pull request to `main`. The other member reviews tests and interfaces. Merge small compatible increments, then sync `main` into your branch:

```bash
git fetch origin
git merge origin/main
# resolve conflicts with the other owner; do not overwrite their work
```

Do not force-push shared branches. Keep datasets, checkpoints, virtual environments, credentials and large artifacts out of Git. Store durable artifacts outside ephemeral GPU runtimes and record their locations/checksums. Exact run evidence belongs in the progress log and experiment manifest.
