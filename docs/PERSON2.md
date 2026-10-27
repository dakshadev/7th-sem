# Person 2 handoff — data, evaluation and execution notebook

Branch: `person2`. Teammate identity unconfirmed. Read AGENTS.md and shared docs first. This role writes core research code, not just slides.

## Owned files

src/data.py, src/evaluate.py, notebooks/run_colab.ipynb, tests/test_data.py, tests/test_evaluation.py, results/ small evidence files. Coordinate dependencies with Person 1. Do not implement a second encoder or change training internals without agreement.

## Ordered checklist — status

- [x] Confirm batch and checkpoint interfaces with Person 1. Send a tiny inspectable batch early.
- [x] Implement explicit WikiText dataset/config/revision selection and a deterministic small pilot subset. Record split and tokenizer manifests; never put raw datasets in Git.
- [x] Tokenize with bert-base-uncased; implement 128-token sequences, padding, special-token exclusion and documented boundary policy.
- [x] Implement independent-token masking, then short-span masking. Ensure approximately comparable eligible-token masking rates and reproducible validation masks.
- [x] Produce real corpus counts, lengths, token-frequency plots and readable masking examples. Do not confuse these statistics with evidence of model collapse.
- [x] Build geometry metrics first using small synthetic matrices, then real student outputs: centered effective rank, variance and centered cosine diagnostics with clear definitions/degenerate-case handling.
- [x] Load Person 1's student checkpoint, freeze it and perform uncorrupted masked-mean pooling excluding special tokens/padding.
- [x] Add frozen AG News linear probe. Record dataset splits/revisions, train-only preprocessing, seed, accuracy, macro-F1 and majority baseline. The random-init encoder reference is reported alongside every probe.
- [x] Create a Colab/Kaggle launcher that installs the tested environment, checks GPU, checks out an explicit code revision and runs the actual supported commands. No secrets embedded; public clone, no token.
- [ ] Save actual logs/plots/tables and durable checkpoint handles. Test recovery after runtime reset, not just an in-session run. The restart cell is now labelled a same-session reload; a true runtime-reset recovery test is still outstanding.

## Metric cautions

Document centering axis, covariance/variance convention, SVD rank definition and sampled pairs. For RankMe-style effective rank state whether singular values or covariance eigenvalues are normalized; these are not interchangeable. Centered pairwise cosine averages alone are not a reliable collapse score; show distributions together with variance, rank and transfer. Zero-norm vectors or all-zero spectra must return an explicit degenerate status, not misleading valid-looking numbers. Rank is bounded by evaluation sample count and embedding dimension; use fixed counts for comparison. No universal collapse threshold is preselected.

## Required tests

Shape/dtype and valid ID range; no special/padding target positions; corruption does not alter unselected tokens; reproducibility by seed; span boundaries; no-target/empty-example handling; source split isolation; pooling excludes padding/special tokens; finite metrics on healthy fixture; degenerate detection on constant fixture; encoder remains unchanged during evaluation; probe preprocessing does not fit on held-out examples. Unit tests use local fixtures, not downloads.

## First acceptance checkpoint

A tiny batch meeting docs/INTERFACES.md and readable original/masked examples. Hand this to Person 1 immediately. Next: corpus statistics and tested evaluation functions while their trainer is being built.
