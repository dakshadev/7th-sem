# Experiment and evidence protocol

This protocol describes what counts as a valid run. The measured runs that have been executed
are tabulated at the end.

## Staged execution

1. CPU unit tests and synthetic smoke runs: correctness only.
2. One real-data MLM step: integration.
3. Short GPU pilot: measure memory, tokens/second, wall time and checkpoint recovery.
4. Fix a feasible common token budget based on actual quota/throughput. Log optimizer updates and batch settings too. Do not assume equal steps imply equal tokens or equal compute.
5. Train C (MLM) and a documented latent configuration; prioritize correctness over breadth. Prefer A and B anchors if time allows.
6. Evaluate fixed held-out representations and initial AG News frozen probes as available.
7. After review: full D-G component switches and key repeated seeds. Optional H/SST-2 last.

## Comparison rules

Use the same backbone architecture, tokenizer, dataset selection and split; align initialization seeds where meaningful. Record differing heads/predictor costs and teacher overhead. Match data-token exposure first and report measured compute separately; do not claim equal compute from equal token counts. Shared optimizer schedule and settings should be explicit; deviations must be justified. A vs B changes multiple components, so it cannot isolate their causal effects.

A lower latent loss is not directly comparable to MLM cross-entropy. Plot separately by objective. Show step and token axes and distinguish training from validation. A short pilot cannot establish final superiority, universal collapse behavior or a confirmed research hypothesis.

## Minimum run manifest

run_id; variant/config path; seed; code commit and dirty status; Python/library versions; device/GPU; dataset and tokenizer revisions/manifests; train/validation selection; nonpadding tokens and masked targets; optimizer steps; elapsed time and timing convention; objective/predictor/teacher settings; checkpoint/log locations; completed/interrupted/failed status; resume limitations.

Use GPU-hours only with an explicit convention (elapsed active training wall time on the stated allocated GPU count), including whether evaluation/setup are excluded. Record interrupted segments and cumulative totals; no estimated cost presented as measured cost.

## Evaluation

- Fixed held-out representation sample; unmasked frozen student; documented pooling and sample count.
- Centered effective rank, per-dimension variance summaries and cosine distributions with degenerate-status reporting. Metrics need definitions and limits, not arbitrary success thresholds.
- AG News frozen linear probe: train only classifier/preprocessing on training labels, report held-out accuracy and macro-F1. Avoid tuning on the final test set.
- Label incomplete probes as not run; never fill table cells with guessed values.
- Two seeds are limited robustness evidence, not a strong basis for narrow confidence intervals. Report individual results and limitations.

## Evidence table

All five groups are measured and stored as JSON in `results/person1/`. Commit, config and
counting conventions for each run are in `docs/PROGRESS.md`; the narrative is in
`docs/PERSON1_LONG_PILOT.md`, `docs/PERSON1_GPU_PILOT.md` and `docs/PERSON1_PROBE.md`.

| Run group | Artifact | What it establishes |
|---|---|---|
| Pilot, 50 updates, WikiText-2 400 records | `results/person1/kaggle_t4_pilot.json` | The pipeline runs on a real GPU and writes comparable artifacts. No design conclusion. |
| Ablations D-G, one change from A, seed 17 | `results/person1/kaggle_long_runs.json` | Local component effects at the 700k-token budget, loss only, single seed. |
| Frozen geometry on held-out WikiText-2 | `results/person1/geometry_long.json` | No condition collapses; MLM learns a markedly lower-rank space than A or B. |
| AG News frozen linear probe | `results/person1/probe_agnews_long.json`, `results/person1/sweep_annealed.json` | No trained student beats a randomly initialised encoder; a 7x larger budget costs 0.04-0.05 accuracy. |
| Objective 1 ambiguity analysis | `results/person1/ambiguity_wikitext2.json` | ~30% of masked positions have several plausible fillers; span masking is not more ambiguous than independent masking at this scale. |

`results/person1/sweep_pass1_no_annealing.json` is superseded and must not be cited as the
budget comparison.

Store small CSV/JSON logs and plots in results/; keep raw text, embeddings and checkpoints
elsewhere. Provide authorized collaborators with accessible durable artifact locations, not
notebook-runtime-only paths.
