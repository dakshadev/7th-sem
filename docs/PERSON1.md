# Person 1 handoff — Daksh: model and training

Branch: `person1`. Read AGENTS.md and shared docs first. Do not implement Person 2's data pipeline as a competing production module. Synthetic fixtures are allowed to remove the dependency on their work.

## Owned files

src/model.py, src/objectives.py, src/train.py, configs/, tests/test_model.py, tests/test_objectives.py, tests/test_training.py. Coordinate dependency/environment files with Person 2 before editing. Shared interface changes must be agreed, not imposed.

## Ordered checklist (all pending at bootstrap)

- [ ] Establish a supported Python/PyTorch environment; record installed versions and tested setup commands. Prefer a simple installable project, not notebook-only code.
- [ ] Define explicit validated config: vocabulary, model dimensions, FFN/dropout/norm settings, optimizer, masks, predictor, teacher targets, EMA, seed and token budget.
- [ ] Implement the shared random-initialized encoder with learned positions, padding-aware attention and intermediate layer access.
- [ ] Add MLM vocabulary head and selected-position cross-entropy.
- [ ] Exercise a synthetic optimization step, then a small fixed-batch overfit diagnostic. Label both as implementation tests.
- [ ] Connect Person 2's batch contract; run one real batch and a short real MLM pilot.
- [ ] Implement teacher copy/freeze/eval, post-step EMA, masked latent targets and detached teacher outputs. Add a simple debug latent configuration if helpful, explicitly separate from recipe A.
- [ ] Implement recipe A faithfully to this project's documented config, including its attention predictor; implement B only after the path works.
- [ ] Add resumable checkpoint save/load and exportable student checkpoint. Coordinate the schema with Person 2.
- [ ] Run actual GPU pilots with logs and metadata. Never promise throughput before measuring it.

## Required tests

Shapes/dtypes; padding-mask semantics; finite gradients/loss; only selected positions contribute to MLM loss; no-target guard; student parameters change after optimization; teacher has no gradients; EMA update matches formula on a tiny fixture; initial teacher equals student; target layer selection is correct; checkpoint roundtrip and resume counters/state. No network needed for unit tests.

## First acceptance checkpoint

A CPU synthetic MLM step executes and demonstrably changes parameters; tests and exact command are recorded. Next: one real batch from Person 2. Do not wait for an entire dataset before testing model code.

## How the next AI should start

Inspect current status and progress, find which checklist items actually exist, run their tests, then implement the earliest missing dependency. Do not regenerate the whole repository or report this handoff as completed code. End each session with changed files, real test output, blockers and the next integration action.
