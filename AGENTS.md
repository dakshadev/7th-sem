# AI continuation instructions

## Read before acting

Read README.md, docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, docs/EXPERIMENT_PROTOCOL.md,
docs/DATA_INTEGRITY.md and docs/PROGRESS.md, then the handoff matching the current branch.
Inspect git status, branch, tracked files and recent commits first.

`main` is the implemented tree, not a plan: the encoder, objectives, data pipeline, training
loop, evaluation and tests exist, the suite is `55 passed, 2 skipped`, and results/ holds
measured runs. The docs describe that work. Still, run the suite before assuming a subsystem
works (`python -m pytest -q`) and check docs/PROGRESS.md for the latest entry, which is the
only place that records what was actually executed and when.

Two documents anchor that evidence: docs/PERSON1_LONG_PILOT.md (measured pilots, ablations,
probe, ambiguity analysis) and docs/DATA_INTEGRITY.md (what an independent audit verified, and
the six ways these artefacts can be misread — seed provenance, two training clocks, a
stream-size field that looks like a token budget, an unpinned random baseline, sub-0.002 probe
noise, and WikiText-2 being a prefix of WikiText-103).

## Role routing

- `person1`: Daksh's model/training work; follow docs/PERSON1.md.
- `person2`: teammate's data/evaluation work; follow docs/PERSON2.md.
- `main`: shared reviewed integration. Implementation belongs on a role branch.
- A different branch or unclear role: resolve ownership with the user before editing shared interfaces.

## Non-negotiables

1. No invented runs, losses, plots, accuracy, progress, literature claims or GPU-hour estimates presented as measured facts.
2. Implement the small encoder from scratch. The pretrained BERT **tokenizer** is allowed; do not silently load pretrained encoder weights. Optional pretrained-BERT ambiguity analysis is separate and must be labelled.
3. Use shared data/configuration and comparable token budgets. A/B are recipe-inspired anchors, not faithful paper reproductions.
4. Add deterministic CPU unit tests for implemented components. No network access in unit tests; separate dataset/GPU integration tests. Exercise code before calling it complete.
5. Loss alone does not establish non-collapse or useful transfer. Report geometry and downstream evidence with limitations.
6. Preserve padding, special-token and train/validation exclusion rules in docs/INTERFACES.md. Never silently change contracts. Coordinate changes with both owners.
7. Do not invent dependency versions or executable commands. Select, install, test and document an actual environment. Keep modules small, typed and readable; no unnecessary framework or UI.
8. Checkpoints should resume training state, not just weights. Record code commit, config, data manifest and environment for every real experiment.
9. Never stage secrets, datasets, generated embeddings, checkpoints or virtual environments. Never force-push, erase another member's work or commit unrelated files.
10. After verified work, update docs/PROGRESS.md: date, branch, actual command/environment, outcome, artifact locations, remaining work and next action. Never turn planned checkboxes into completed ones without evidence.

## Delivery standard

A module is done when its interface works, tests pass and a real smoke invocation exercises it. A training result is done when its config, logs and checkpoint are recoverable. A research conclusion is provisional until supported by adequate controlled runs. Do not spend the remaining pre-review time building optional ablations while core integration is broken.
