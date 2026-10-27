# AI continuation instructions

## Read before acting

Read README.md, docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, docs/EXPERIMENT_PROTOCOL.md and docs/PROGRESS.md, then the handoff matching the current branch. Inspect git status, branch, tracked files and recent commits first. These documents are a handoff, not proof that implementation exists.

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
