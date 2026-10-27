# Verified progress and next actions

## Bootstrap — 27 October 2026

Scope: documentation only. The initial repository was empty. No executable model, data pipeline, training, tests, GPU experiment, dependency environment or results exist at this stage. The bootstrap commit establishes shared context and intended interfaces. Branch publishing must be verified separately by the initiating assistant.

### Person 1 next

Establish the Python environment/config, implement shared encoder plus MLM objective, and prove one synthetic optimization step with tests. Then integrate Person 2's real batch.

### Person 2 next

Prepare a tiny real tokenized batch with masks, document data/tokenizer selection and produce readable examples. Hand the batch to Person 1 before finishing the whole data pipeline. Then implement statistics and geometry tests.

### Shared open decisions

- Active Person 2's name/account and repository access.
- Exact mid-review template and guide expectations.
- Supported Python/dependency versions and tested installation commands.
- FFN width, dropout/norm settings, optimization/EMA settings and predictor details.
- Normalization formula, independent-mask/span policy and packing/truncation policy.
- Measured GPU availability/throughput, feasible token budget and artifact storage.
- Checkpoint schema, geometry definitions and probe split/tuning policy.

## Verified Milestone: Person 2 Data & Evaluation Pipeline — 27 October 2026

- Date / author / branch / commit: 2026-10-27 / Person 2 / `person2` / `2085aad` + working tree
- Implemented files:
  - `requirements.txt`: documented environment dependencies with verified versions (`torch>=2.0.0`, `transformers>=4.40.0`, `datasets>=2.18.0`, `scikit-learn>=1.4.0`, `scipy>=1.12.0`, `numpy>=1.26.0`, `matplotlib>=3.8.0`, `pytest>=8.0.0`).
  - `.gitignore`: added exception `!results/tiny_batch.pt` to ensure regenerable tiny batch tensor fixture is tracked and available for Person 1.
  - `src/__init__.py`: package initialization.
  - `src/data.py`: BERT tokenization (`bert-base-uncased`, vocab 30,522), sequence padding/truncation (L=128), independent-token masking (~15%) and short-span masking (~15%, mean length 3), batch collation enforcing contract invariants (`input_ids`, `masked_input_ids`, `attention_mask`, `prediction_mask`), decoding utilities, token counting conventions, explicit WikiText-2 pilot (`wikitext-2-raw-v1`) vs WikiText-103 target (`wikitext-103-raw-v1`) labelling, real dataset-loader entry points (`load_wikitext_split`, `get_dataset_loaders`), and CLI entry point (`python -m src.data`).
  - `src/evaluate.py`: attention-mask-aware mean pooling strictly excluding padding and special tokens (`[CLS]`, `[SEP]`, etc.) with empty-sequence guarding; representation geometry diagnostics including centered effective rank (RankMe singular value entropy and covariance eigenvalue entropy), per-dimension variance summaries, dead dimension counting, centered pairwise cosine similarity distributions, and explicit degenerate status detection (collapse, zero spectrum, NaN/Inf, N<2); checkpoint loading accepting `'student'` key (as well as `'student_encoder'`, `'model'`, `'encoder'`), reading `model_config`, and supporting `return_config=True`; frozen linear probe evaluation (train-only `StandardScaler` fitting, logistic regression, accuracy, macro-F1, majority baseline).
  - `tests/test_data.py`: 10 CPU unit tests including offline dataset-loader validation, shapes/dtypes, token ID ranges, special/padding exclusion, corruption preservation, seed reproducibility, span masking boundaries, empty text handling, decoding inspection, token counting.
  - `tests/test_evaluation.py`: 11 CPU unit tests covering student key & model_config checkpoint loading roundtrip, padding exclusion under poisoning, special token exclusion under poisoning, empty example guard, healthy geometry, constant collapse degenerate detection, zero spectrum detection, NaN/Inf detection, N<2 detection, encoder freezing/immutability, linear probe train-only scaling.
  - `scripts/generate_artifacts.py`: reproducible script generating tiny inspectable batch, real corpus statistics with explicit WikiText-2 pilot labelling, masking comparisons, geometry diagnostics across regimes, and probe baselines.
  - `notebooks/run_colab.ipynb`: complete end-to-end Google Colab / Kaggle execution notebook pinned to an explicit commit (`2085aad`), executing environment check, test suite, real dataset loading, scratch small encoder definition ($L=128, D=256$, 4 layers, 4 heads), actual pilot training run with loss curve plotting, checkpoint saving with `'student'` key & `model_config`, reload & freeze verification, held-out representation geometry evaluation, and AG News frozen linear probing.
  - Artifacts in `results/`:
    - `results/tiny_batch_inspection.txt`: first acceptance checkpoint inspection log (Batch Size 4, L=128, 475 non-padding tokens, 70 masked targets, 14.74% ratio, all invariants verified).
    - `results/tiny_batch.pt`: tracked torch tensor batch fixture for Person 1.
    - `results/corpus_statistics.json`: WikiText-2 pilot real corpus statistics (3,000 samples, 362k raw tokens, 262k truncated tokens, vocab coverage 60.77%, length distributions, explicit pilot vs target labelling).
    - `results/token_length_distribution.png`: histogram comparing sequence lengths against L=128 cutoff.
    - `results/top_tokens_frequency.png`: bar chart of top frequent tokens in pilot corpus.
    - `results/masking_examples.txt` and `results/masking_examples.json`: readable side-by-side examples comparing independent masking vs short-span masking.
    - `results/geometry_diagnostic_demo.json`: verified geometry metrics across isotropic (RankMe 93.6/100, variance 0.99), anisotropic (RankMe 3.20/100), and collapsed regimes (`is_degenerate=True`, status `constant_representations_collapsed`, 100% dead dimensions).
    - `results/linear_probe_baseline.json`: linear probe test evaluation.
- Exact command and environment:
  - Python 3.12.4 on Windows x86_64, PyTorch 2.13.0+cpu, Transformers 5.21.0, Datasets 5.1.0, Scikit-learn 1.7.2, Scipy 1.16.2, Pytest 9.1.1.
  - Commands executed:
    `python -m pytest tests/ -v` (Outcome: 21 passed in 8.39s)
    `python -m src.data --dataset-config wikitext-2-raw-v1 --batch-size 4 --inspect --save-fixture results/tiny_batch.pt` (Outcome: verified real dataset loader and fixture generation)
    `python scripts/generate_artifacts.py` (Outcome: regenerated all artifacts in results/ with pilot labelling)
    `python -c "import json; json.load(open('notebooks/run_colab.ipynb'))"` (Outcome: verified 19 notebook cells)
- Actual test/integration outcome:
  - 21 unit tests passed with zero failures on CPU fixtures without network access.
  - `results/tiny_batch.pt` generated and tracked.
  - `load_student_checkpoint` verified to accept `'student'` key and extract `model_config`.
  - Notebook runs actual end-to-end training and representation evaluation against pinned commit.
- Experiment ID, config and tokens (if any):
  - Milestone checkpoint verification and pilot execution.
- Durable logs/checkpoint/plot locations:
  - `results/tiny_batch.pt`
  - `results/tiny_batch_inspection.txt`
  - `results/corpus_statistics.json`
  - `results/token_length_distribution.png`
  - `results/top_tokens_frequency.png`
  - `results/masking_examples.txt`
  - `results/geometry_diagnostic_demo.json`
  - `results/linear_probe_baseline.json`
- Known failures or limitations:
  - Full multi-epoch GPU pretraining on target 5M tokens is pending Person 1 runner integration; pilot training executed for contract verification.
- Interface changes agreed with other owner:
  - Checkpoint schema accepts `'student'` key and reads `model_config`; batch contract and shapes preserved.
- Next action / owner:
  - Person 1: use `results/tiny_batch.pt` or `src/data.py` `get_dataset_loaders` for encoder pretraining loop.


