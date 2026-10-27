# Person 1 execution plan — Daksh

Planning baseline: 27 October 2026. Mid-project review: 29 October 2026. Repository currently contains documentation only; no implementation milestone below is completed. This is a proposed execution plan, not test evidence or a promise of training throughput.

Read AGENTS.md, PROJECT_CONTEXT.md, INTERFACES.md and PERSON1.md first. Work on branch `person1`. Do not change Person 2's modules or shared contracts without agreement. Local workspace: `C:\Users\daksh\MyLife\Work\7th-sem`.

## Definition of Person 1 complete

A tested shared encoder supports MLM C, latent recipe A, and recipe B; one runner can train/validate each using explicit config; teacher/EMA and selected-target losses are tested; training can save/resume with documented reproducibility limits; the frozen student can be loaded independently by Person 2; actual GPU pilots have recorded configuration, logs, checkpoint handles, tokens and measured time. Full ablation runs and optional SIGReg are later project work, not prerequisites to the first review demo.

## Phase 0 — coordination and scope

- [ ] Ask Person 2 to confirm INTERFACES.md and deliver a tiny real batch early.
- [ ] Confirm who owns dependency files; propose Person 1 initializes the environment/config and Person 2 adds only agreed dependencies.
- [ ] Agree on student export schema before implementing checkpoint consumers.
- [ ] Agree on durable artifact location and collaborator access. No tokens in notebook cells or repository files.
- [ ] Keep the proposed architecture: random initialization, 4 layers, width 256, 4 heads, length 128; verify tokenizer vocabulary from metadata when real data arrives.

Exit gate: both members know the batch keys, output shape and file ownership. Do not wait for their entire pipeline to start synthetic tests.

## Phase 1 — environment and configuration

Deliver: a supported tested Python environment, project/dependency metadata, validated configuration and CPU test entry point.

- [ ] Inspect actual local interpreter availability and Colab environment before choosing versions. Do not assume the host's default Python is compatible with PyTorch.
- [ ] Establish and test isolated local environment; record dependencies and installation command. Check Colab compatibility separately; GPU is for training, not a requirement for CPU unit tests.
- [ ] Define architecture settings, tokenizer/vocabulary metadata, optimizer/schedule, batch and token budget, masking policy reference, predictor, teacher target, EMA, seed, precision and output paths.
- [ ] Make unspecified proposal settings explicit: FFN width, dropout, activation, normalization order, embedding normalization, weight tying if any, optimizer learning rate/decay and EMA schedule.
- [ ] Validate invalid dimensions, unsupported settings and inconsistent configuration early.

Exit gate: a fresh environment can import the project, load valid config and run a trivial test. No training command is considered supported until actually exercised.

## Phase 2 — shared encoder

Deliver: src/model.py and model tests.

- [ ] Implement token and position embeddings and shared encoder stack.
- [ ] Preserve correct padding-mask polarity at the attention API boundary.
- [ ] Return last_hidden_state and optionally the four post-layer hidden states using the shared interface.
- [ ] Use random weights, not a pretrained encoder. Separate encoder from MLM/prediction heads.
- [ ] Test shapes, optional hidden states, invalid input handling and finite forward/backward outputs.
- [ ] In evaluation mode, test that changing padded-token IDs does not change valid-token outputs, holding masks/length fixed.

Exit gate: synthetic masked token IDs produce expected token embeddings and gradients without network/data downloads.

## Phase 3 — MLM objective and synthetic training

Deliver: src/objectives.py MLM path, initial src/train.py and tests.

- [ ] Add vocabulary head and compute cross-entropy only on prediction_mask positions, with original input_ids as labels.
- [ ] Handle batches with no targets without NaNs; do not advance training counters as if a skipped update happened.
- [ ] Add seed control, optimizer step, basic logging and finite-value checks.
- [ ] Run one synthetic CPU step and assert actual parameter change.
- [ ] Run a small fixed-batch overfit diagnostic; investigate failure rather than imposing a guessed exact loss threshold.
- [ ] Test that unselected label values do not affect the masked loss, and padding/special tokens cannot be selected by valid batches.

Exit gate: recorded commands/tests prove the training path works. Synthetic token memorization is correctness evidence only.

## Phase 4 — real MLM integration and early checkpointing

Dependency: Person 2 supplies one real batch; begin as soon as available.

- [ ] Consume their batch contract without a competing production loader.
- [ ] Verify IDs, special-token metadata, masks and actual eligible/predicted token counts together.
- [ ] Run one real CPU/GPU step, then a short GPU throughput/memory pilot.
- [ ] Save/reload an early checkpoint and export a student encoder for Person 2 before long runs.
- [ ] Add validation mode and verify evaluation does not update weights; separate train/validation logs.
- [ ] Select the feasible subsequent token budget based on actual GPU behavior/quota, not an invented duration.

Exit gate: real MLM log, checkpoint and independently loadable student export. Handoff these to Person 2 so evaluation can proceed in parallel.

## Phase 5 — latent core and EMA correctness

Deliver: frozen teacher, target builder, predictor interface and latent training step.

- [ ] Copy the initialized student encoder to teacher; disable teacher gradients and keep target computation in evaluation/no-gradient mode.
- [ ] Send unmasked IDs to teacher, masked IDs to student; select the same prediction positions.
- [ ] Add explicit target builder and predictor; detach teacher targets.
- [ ] Update teacher by EMA after a successful optimizer step only, including correct behavior for skipped AMP/invalid steps if supported.
- [ ] Test teacher/student initial equality, absence of teacher gradients, detached targets, exact toy EMA update and finite latent loss/gradients.
- [ ] A debug linear-predictor latent step may precede A, but name it debug_latent rather than mislabelling it recipe A.

Exit gate: a synthetic latent update changes the student through prediction loss; teacher changes only through the expected EMA rule. Low loss does not establish non-collapse.

## Phase 6 — actual A and B configurations

- [ ] A: final-layer teacher targets, short-span masks, 2-layer attention predictor, MSE. Confirm Person 2's span implementation exists before claiming A ran.
- [ ] Define the attention predictor's architecture, positional information and input sequence explicitly. Coordinate valid-mask handling. Do not accidentally give the predictor teacher/unmasked inputs.
- [ ] B: mean of normalized top-2 teacher layers, independent-token masks, linear predictor, Smooth L1.
- [ ] Specify B's normalization type/axes, order of normalization and averaging, loss reduction and Smooth L1 settings. Test top-2 refers to actual encoder layers, not embeddings.
- [ ] Verify C/A/B share encoder/data/token-budget settings where appropriate; record intentional differences.

Exit gate: each selected recipe has an explicit config and tested objective/forward path. Config files alone do not count as trained variants.

## Phase 7 — robust runner and checkpoint/resume

Start essentials in Phase 4; complete hardening before relying on unattended GPU sessions.

- [ ] One runner supports C/A/B without copied training scripts.
- [ ] Log objective name, steps, nonpadding input tokens, masked targets, elapsed time and failures. Record commit, dirty state, environment, hardware, config and data manifests.
- [ ] Save student, heads/predictor, teacher when present, optimizer, scheduler/AMP state when present, counters, RNG and data/sampler resume state or explicit limits.
- [ ] Version checkpoint format; make student export loadable without the training objective/teacher.
- [ ] Test save/load and next-step continuation on a deterministic CPU fixture. Verify teacher remains frozen/eval after loading.
- [ ] Reject incompatible resume configurations. Preserve existing checkpoints on interrupted writes where practical.
- [ ] Verify checkpoint survives runtime reset via the agreed durable store; a runtime-only path is not a safe handoff.

Exit gate: interrupted training can resume with documented exactness/limitations and Person 2 can load the encoder independently.

## Phase 8 — measured GPU pilots and research handoff

- [ ] Run C and at least one correctly labelled latent configuration; A/B anchors preferred when capacity allows.
- [ ] Use a recorded common data-token budget for comparisons. Report different predictor/teacher overhead rather than claiming identical compute.
- [ ] Send Person 2 checkpoint export, config, run ID, data manifest and logs for fixed-sample geometry and frozen probes.
- [ ] Record OOM/quota interruptions and partial runs honestly; do not fill missing result cells.
- [ ] Verify all tests and supported commands. Update PERSON1.md checkboxes only for tested work, and append actual evidence to PROGRESS.md.
- [ ] Commit focused changes on person1; use reviewed PRs to main. Never overwrite Person 2's work or push implementation straight to main.

Exit gate: real run evidence is accessible and another member can reproduce the training/evaluation handoff.

## Deadline prioritization

**27 October:** coordination, environment/config, encoder, synthetic MLM, real-batch integration as soon as available; begin latent core when MLM works. Integrate early rather than polishing standalone modules.

**28 October:** complete tested latent path and checkpoint recovery, run feasible real pilots, hand evidence to Person 2, verify demo and document limitations. B is desirable but must not displace a working C plus one honest latent run.

**29 October:** present only verified progress. Avoid last-minute architecture changes. Demonstrate saved artifacts if GPU access fails.

After the review: finish incomplete A/B configurations and runner hardening, full component switches, longer runs and seed repeats. SIGReg remains optional. A correct, explained partial milestone is preferable to claiming all phases completed without evidence.

## Prompt for the next implementation session

> Work in C:\Users\daksh\MyLife\Work\7th-sem on person1. Read AGENTS.md, shared context/interfaces/protocol, PERSON1.md, PERSON1_PLAN.md and PROGRESS.md. Inspect actual code/status. Implement the earliest unfinished phase, with tests and a real smoke invocation. Keep Person 2's ownership intact. Start with environment/config plus encoder and synthetic MLM if nothing exists. Record exact commands/results and stop at a coherent tested checkpoint, not a stub. Do not push to main or claim experiments that were not run.
