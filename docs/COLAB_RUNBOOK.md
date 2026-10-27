# Free Colab / Kaggle runbook — Person 1 pilots

Kaggle T4 runs were executed on 27 October 2026. The measured numbers are in
`docs/PROGRESS.md`, `docs/PERSON1_GPU_PILOT.md` and `docs/PERSON1_LONG_PILOT.md`. This file is
the procedure for reproducing and extending them, not a pending plan. Do not mark a checkbox,
table cell or result as complete until the command actually ran and its output was captured.

The project pins Python `>=3.12,<3.13` and `torch==2.9.1` in `pyproject.toml` for local CPU
work. Person 2's `requirements.txt` documents `transformers`, `datasets`, `scikit-learn`,
`scipy`, `numpy`, `matplotlib` and `pytest` bounds. Do not invent a different CUDA build
locally.

## 0. Prerequisite: Person 2's modules must be present

`scripts/build_batches.py`, `scripts/run_pilot.py` and `src/evaluate_bridge.py` deliberately
do **not** vendor `src/data.py` or `src/evaluate.py`; they import them lazily and fail with
an actionable message when absent. `main` is the canonical integrated branch and contains both
modules, so clone it (section 2). Do not merge or edit another owner's branch unilaterally.

```bash
python -c "import src.data, src.evaluate; print('Person 2 modules present')"
```

## 1. Select the runtime

- Colab: **Runtime → Change runtime type → T4 GPU** (free tier).
- Kaggle: **Notebook options → Accelerator → GPU T4 ×2 or P100**, and enable **Internet**.

Verify the GPU before doing anything else:

```python
!nvidia-smi
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
assert torch.cuda.is_available(), "No CUDA device; stop and fix the runtime selection"
```

If `torch.cuda.is_available()` is false, stop. A CPU-only cloud run is not a GPU result.

## 2. Clone the public repository

The repository is public, so the clone needs no token. Clone `main`, the canonical integrated
branch, then pin a commit for reproducibility.

```bash
cd /content
git clone https://github.com/dakshadev/7th-sem.git
cd 7th-sem
```

`--depth 1` is acceptable only if you pin to the fetched tip. For an arbitrary older commit:

```bash
git fetch --unshallow || true
```

Pin the exact commit (paste the SHA from the verified push report):

```bash
COMMIT="<paste-the-main-commit-sha>"
git checkout --detach "$COMMIT"
test "$(git rev-parse HEAD)" = "$COMMIT" && echo "pinned at $COMMIT"
```

## 3. Install dependencies

Check the runtime Python and existing Torch first. Colab/Kaggle usually ship a CUDA-enabled
Torch.

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The project requires Python 3.12. If the runtime reports a different minor version, stop and
select a compatible image rather than silently relaxing `requires-python`.

Use the CUDA build the image already ships. Never install the CPU pin (`torch==2.9.1` from
`pyproject.toml`, or `torch` from `requirements.txt`) on a GPU image: it replaces the
accelerated build with a CPU wheel and every later step silently runs on CPU.

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA build missing; fix the image, do not pip install torch'"
```

Only if Torch is genuinely absent, install a CUDA wheel from the PyTorch index matching that
image's CUDA version. Do not add a CPU index on a GPU runtime.

Then install the project without replacing Torch, plus Person 2's documented dependencies:

```bash
pip install -e . --no-deps
pip install "transformers>=4.40.0" "datasets>=2.18.0" "scikit-learn>=1.4.0" \
  "scipy>=1.12.0" "numpy>=1.26.0" "matplotlib>=3.8.0"
```

Local CPU development keeps the CPU-only wheel and is **not** the cloud path:

```bash
uv pip install --python .venv/Scripts/python.exe "torch==2.9.1" \
  --index-url https://download.pytorch.org/whl/cpu
```

## 4. Run the test suite in the cloud runtime

```bash
cd /content/7th-sem
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

Expect **55 passed, 2 skipped**. One skip is the CUDA-only resume test, the other is an
integration-only test. On a working CUDA runtime the resume test should execute, so record the
real output rather than assuming a count.

## 5. Choose a durable artifact directory

```bash
if [ -d /content/drive/MyDrive ]; then
  ARTIFACTS=/content/drive/MyDrive/7th-sem-artifacts
else
  ARTIFACTS=/kaggle/working/7th-sem-artifacts
fi
mkdir -p "$ARTIFACTS"
echo "$ARTIFACTS"
```

In Colab, mount Drive first:

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 6. Build batch artifacts at the pilot budget

Independent masking (used by `mlm` and `latent_b`):

```bash
cd /content/7th-sem
python scripts/build_batches.py \
  --dataset Salesforce/wikitext \
  --config-name wikitext-2-raw-v1 \
  --split train \
  --tokenizer bert-base-uncased \
  --max-length 128 \
  --policy independent \
  --mask-ratio 0.15 \
  --max-records 12000 \
  --batch-size 8 \
  --seed 17 \
  --out "$ARTIFACTS/batches_independent.pt"
```

Short-span masking (used by `latent_a`):

```bash
python scripts/build_batches.py \
  --dataset Salesforce/wikitext \
  --config-name wikitext-2-raw-v1 \
  --split train \
  --tokenizer bert-base-uncased \
  --max-length 128 \
  --policy span \
  --mask-ratio 0.15 \
  --max-records 12000 \
  --batch-size 8 \
  --seed 17 \
  --out "$ARTIFACTS/batches_span.pt"
```

For the 5M-token tier use `--config-name wikitext-103-raw-v1` with `--max-records 90000`.
WikiText-2 train is a prefix of WikiText-103 train, so at a 700k-token budget the two labels
describe the same data. Do not present them as different corpora at that budget.

Each run writes `*.manifest.json` next to the artifact. Inspect `artifact.sha256`, `source`,
`tokenizer.revision`, `counts` and `limitations`. The manifest states explicitly that this is a
bounded subset, **not** the full proposed 5M-token budget. Only one split is read; train and
validation are never joined. Do not place raw datasets or the generated tensor artifacts inside
the repository.

Offline smoke without a dataset download (create a throwaway fixture first):

```bash
cat > /tmp/tiny_texts.json <<'JSON'
{"texts": [
  "a small encoder learns masked latent targets",
  "representation geometry reports effective rank",
  "the pilot records measured tokens and wall clock time"
]}
JSON
python scripts/build_batches.py --offline-fixture /tmp/tiny_texts.json \
  --tokenizer bert-base-uncased --max-length 128 --max-records 3 --batch-size 2 \
  --seed 17 --out "$ARTIFACTS/batches_fixture.pt"
```

## 7. Run the three pilots to a token budget

`run_pilot.py` loads the checked-in config, reuses or builds the manifested artifact, and
invokes the real `src.train.run_batches`/`Trainer` implementation. Drive the run with
`--max-tokens`: the trainer stops with `stop_reason: budget` once the token budget is reached,
which is what the recorded runs did. Sequence length 128, batch 8 and mask ratio 0.15 come from
the checked-in config; pass them explicitly only when overriding.

```bash
python scripts/run_pilot.py \
  --variant mlm --config configs/mlm_c.json \
  --batches "$ARTIFACTS/batches_independent.pt" \
  --output "$ARTIFACTS/pilot-mlm" \
  --max-tokens 700482 --device auto

python scripts/run_pilot.py \
  --variant latent_a --config configs/latent_a.json \
  --batches "$ARTIFACTS/batches_span.pt" \
  --output "$ARTIFACTS/pilot-latent-a" \
  --max-tokens 700482 --device auto

python scripts/run_pilot.py \
  --variant latent_b --config configs/latent_b.json \
  --batches "$ARTIFACTS/batches_independent.pt" \
  --output "$ARTIFACTS/pilot-latent-b" \
  --max-tokens 700482 --device auto
```

Repeat with `--seed 43` for the second seed, and with `--max-tokens 5000000` for the full-budget
tier. Let the learning-rate schedule anneal to zero inside each run by deriving `max_updates`
from the budget; a fixed cap confounds budget with schedule (see the superseded
`results/person1/sweep_pass1_no_annealing.json`).

**Smoke path only:** `--max-updates 50` is a segment cap that stops a run after 50 updates
regardless of tokens and leaves `stop_reason: interrupted`. Use it to prove the pipeline runs,
never to produce a result.

Each output directory contains `metadata.json`, `metrics.jsonl`, `checkpoint.pt`, `student.pt`
and `pilot.json`. All values in `pilot.json` are measured on that run; fields that were not
measured are `null` or explicitly listed under `not_run`. A bounded segment can leave
`metadata.json`'s lifecycle as `interrupted`; that is the segment cap, not a completed
scientific run. Record machine, Torch/CUDA versions, exact command, token counts, wall-clock
time and checkpoint hashes.

## 8. Export the student

`run_pilot.py` already writes `student.pt`. To re-export from a saved training checkpoint with
the real trainer:

```bash
python - <<'PY'
from pathlib import Path
from src.config import load_config
from src.train import Trainer

checkpoint = Path("/content/drive/MyDrive/7th-sem-artifacts/pilot-mlm/checkpoint.pt")
output = checkpoint.with_name("student-reexport.pt")
trainer = Trainer(load_config("configs/mlm_c.json"), device="cpu")
trainer.load_checkpoint(checkpoint)
trainer.export_student(output)
print(output)
PY
```

The export is a standalone `{"schema_version": 1, "model_config": ..., "student": ...}`
payload; it contains no teacher, head, predictor or optimizer.

## 9. Run frozen representation geometry

Use the held-out `validation` split. Never fit or mask on it, and never join it to `train`.

```bash
python -m src.evaluate_bridge \
  --student "$ARTIFACTS/pilot-mlm/student.pt" \
  --dataset Salesforce/wikitext \
  --config-name wikitext-2-raw-v1 \
  --split validation \
  --tokenizer bert-base-uncased \
  --max-length 128 \
  --max-records 300 \
  --batch-size 8 \
  --device auto \
  --out "$ARTIFACTS/geometry-mlm.json"
```

Repeat with `pilot-latent-a/student.pt` and `pilot-latent-b/student.pt`. The bridge uses the
frozen student in evaluation mode under `torch.no_grad()`, uncorrupted sequences, and Person 2's
attention-mask-aware mean pooling that excludes padding and special tokens. It prints and
writes named geometry metrics. Metrics that were not collected (token geometry, GPU throughput
benchmark, pretrained-BERT analysis) appear under `not_run`; never fill them with invented
values. A `status: "degenerate"` result is a valid measured outcome and must be reported as
such.

## 10. Store durable artifacts

Cloud runtime disks are ephemeral. Copy the whole run to Drive or Kaggle output and record
checksums:

```bash
tar -czf "$ARTIFACTS/7th-sem-pilot-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
  "$ARTIFACTS/batches_independent.manifest.json" \
  "$ARTIFACTS/batches_span.manifest.json" \
  "$ARTIFACTS/pilot-mlm" \
  "$ARTIFACTS/pilot-latent-a" \
  "$ARTIFACTS/pilot-latent-b" \
  "$ARTIFACTS/geometry-mlm.json"
sha256sum "$ARTIFACTS"/*.tar.gz
```

Do not commit datasets, checkpoints, embeddings or `.venv`. Put the durable location and the
manifest/checkpoint hashes in `docs/PROGRESS.md` only after the run actually completes.
