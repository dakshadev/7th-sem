# Lightweight Self-Supervised Text Representation Learning

**Project:** A Lightweight Self-Supervised Framework for Text Representation Learning Using Masked Latent Prediction.

This undergraduate research project investigates how target construction, masking, predictor design and regression loss affect collapse and transfer in a small text encoder. It compares masked latent prediction against a masked-language-modeling (MLM) control.

## Current status — 27 October 2026

Person 1's random small encoder, C/A/B objectives, fp32 runner, resumable checkpoints and export API are CPU-verified on `person1`: 25 deterministic CPU tests passed and one cloud-only CUDA resume test was skipped in the CPU environment. Two-update synthetic C/A/B diagnostics also completed on CPU; they are implementation checks, not research results. Person 2's data/evaluation pipeline, real-data integration and all cloud GPU runs remain pending. See [Person 1 implementation](docs/PERSON1_IMPLEMENTATION.md), [development evidence](docs/DEVELOPMENT_EVIDENCE.md), and [execution environment](docs/EXECUTION_ENVIRONMENT.md).

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

The supported target interpreter is Python 3.12. The verified local CPU environment is PyTorch 2.9.1+cpu, pytest 9.1.1 and ruff 0.14.14; cloud GPU commands remain explicitly unexecuted.

## Collaboration

Work on your assigned branch. Commit focused changes and push your branch, then open a pull request to `main`. The other member reviews tests and interfaces. Merge small compatible increments, then sync `main` into your branch:

```bash
git fetch origin
git merge origin/main
# resolve conflicts with the other owner; do not overwrite their work
```

Do not force-push shared branches. Keep datasets, checkpoints, virtual environments, credentials and large artifacts out of Git. Store durable artifacts outside ephemeral GPU runtimes and record their locations/checksums. Exact run evidence belongs in the progress log and experiment manifest.
