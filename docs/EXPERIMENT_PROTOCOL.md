# Experiment and evidence protocol

This is a proposed protocol, not a record of executed experiments.

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

## Evidence table (populate only with real outputs)

| Run | Commit/config | Tokens | Loss objective | Rank/variance | Probe Acc/F1 | Measured GPU time | Artifacts |
|---|---|---|---|---|---|---|---|

No experiments completed at bootstrap. Store small CSV/JSON logs and plots in results/; keep raw text, embeddings and checkpoints elsewhere. Provide authorized collaborators with accessible durable artifact locations, not notebook-runtime-only paths.
