# Free Colab / Kaggle runbook — Person 1 pilots

**Execution status: no GPU or cloud run has been executed.** Every GPU/cloud step
below is pending the user. The only verified commands are the CPU checks recorded in
`docs/DEVELOPMENT_EVIDENCE.md` and `docs/PROGRESS.md`. Do not mark a checkbox, table cell
or result as complete until the command actually ran and its output was captured.

The project pins Python `>=3.12,<3.13` and `torch==2.9.1` in `pyproject.toml`. Person 2's
`requirements.txt` documents `transformers`, `datasets`, `scikit-learn`, `scipy`, `numpy`,
`matplotlib` and `pytest` bounds. Do not invent a different CUDA build locally.

## 0. Prerequisite: Person 2's modules must be present

`scripts/build_batches.py`, `scripts/run_pilot.py` and `src/evaluate_bridge.py` deliberately
do **not** vendor `src/data.py` or `src/evaluate.py`; they import them lazily and fail with
an actionable message when absent. The `person1` branch does not contain those files. Before
a cloud run, use a reviewed integration state that contains both modules (for example, after
coordinating a merge of `origin/person2` into a scratch integration branch — do not merge or
edit another owner's branch unilaterally).

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

## 2. Clone the private repository

Read the token without echoing it into the notebook output:

```python
from getpass import getpass
import os
os.environ["GH_TOKEN"] = getpass("GitHub token with repo read access: ")
print("token length:", len(os.environ["GH_TOKEN"]))
```

```bash
cd /content
test -n "$GH_TOKEN" && echo "token present"
git clone --branch person1 \
  "https://x-access-token:${GH_TOKEN}@github.com/dakshadev/7th-sem.git" 7th-sem
cd 7th-sem
git remote set-url origin https://github.com/dakshadev/7th-sem.git
```

`--depth 1` is acceptable only if you pin to the fetched tip. For an arbitrary older commit:

```bash
git fetch --unshallow || true
```

Pin the exact commit (paste the SHA from the verified push report):

```bash
COMMIT="<paste-the-person1-commit-sha>"
git checkout --detach "$COMMIT"
test "$(git rev-parse HEAD)" = "$COMMIT" && echo "pinned at $COMMIT"
```

Never write the token into a notebook, a commit, or a logged command. Revoke it after the run.

## 3. Install dependencies

Check the runtime Python and existing Torch first. Colab/Kaggle usually ship a CUDA-enabled
Torch; do not replace a working CUDA build with a CPU wheel.

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The project requires Python 3.12. If the runtime reports a different minor version, stop and
select a compatible image rather than silently relaxing `requires-python`.

If Torch is missing, install the pinned version (the default Linux wheel is the runtime's
CUDA build; do not add a CPU index on the GPU runtime):

```bash
pip install "torch==2.9.1"
python -c "import torch; assert torch.cuda.is_available()"
```

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

The local CPU environment reports **31 passed, 1 skipped**; the skip is the CUDA-only resume
test. On a working CUDA runtime that test should execute (expect 32 passed only if it
actually passes; record the real output).

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

## 6. Build bounded batch artifacts

Independent masking (used by `mlm` and `latent_b`):

```bash
cd /content/7th-sem
python scripts/build_batches.py \
  --dataset Salesforce/wikitext \
  --config-name wikitext-103-raw-v1 \
  --split train \
  --tokenizer bert-base-uncased \
  --max-length 128 \
  --policy independent \
  --mask-ratio 0.15 \
  --max-records 500 \
  --batch-size 8 \
  --seed 17 \
  --out "$ARTIFACTS/batches_independent.pt"
```

Short-span masking (used by `latent_a`):

```bash
python scripts/build_batches.py \
  --dataset Salesforce/wikitext \
  --config-name wikitext-103-raw-v1 \
  --split train \
  --tokenizer bert-base-uncased \
  --max-length 128 \
  --policy span \
  --mask-ratio 0.15 \
  --max-records 500 \
  --batch-size 8 \
  --seed 17 \
  --out "$ARTIFACTS/batches_span.pt"
```

Each run writes `*.manifest.json` next to the artifact. Inspect `artifact.sha256`,
`source`, `tokenizer.revision`, `counts` and `limitations`. The manifest states explicitly
that this is a bounded pilot subset, **not** the proposed 5M-token WikiText-103 budget.
Only one split is read; train and validation are never joined. Do not place raw datasets or
the generated tensor artifacts inside the repository.

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

## 7. Run the three bounded pilots

`run_pilot.py` loads the checked-in config, reuses or builds the manifested artifact, and
invokes the real `src.train.run_batches`/`Trainer` implementation. `--max-updates` is a
segment cap; it does not change the configured learning-rate schedule.

```bash
python scripts/run_pilot.py \
  --variant mlm --config configs/mlm_c.json \
  --batches "$ARTIFACTS/batches_independent.pt" \
  --output "$ARTIFACTS/pilot-mlm" \
  --max-updates 50 --device auto

python scripts/run_pilot.py \
  --variant latent_a --config configs/latent_a.json \
  --batches "$ARTIFACTS/batches_span.pt" \
  --output "$ARTIFACTS/pilot-latent-a" \
  --max-updates 50 --device auto

python scripts/run_pilot.py \
  --variant latent_b --config configs/latent_b.json \
  --batches "$ARTIFACTS/batches_independent.pt" \
  --output "$ARTIFACTS/pilot-latent-b" \
  --max-updates 50 --device auto
```

Each output directory contains `metadata.json`, `metrics.jsonl`, `checkpoint.pt`,
`student.pt` and `pilot.json`. All values in `pilot.json` are measured on that run; fields
that were not measured are `null` or explicitly listed under `not_run`. A bounded segment
can leave `metadata.json`'s lifecycle as `interrupted`; that is the segment cap, not a
completed scientific run. Record machine, Torch/CUDA versions, exact command, token counts,
wall-clock time and checkpoint hashes.

## 8. Export the student

`run_pilot.py` already writes `student.pt`. To re-export from a saved training checkpoint
with the real trainer:

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
  --config-name wikitext-103-raw-v1 \
  --split validation \
  --tokenizer bert-base-uncased \
  --max-length 128 \
  --max-records 500 \
  --batch-size 8 \
  --device auto \
  --out "$ARTIFACTS/geometry-mlm.json"
```

Repeat with `pilot-latent-a/student.pt` and `pilot-latent-b/student.pt`. The bridge uses the
frozen student in evaluation mode under `torch.no_grad()`, uncorrupted sequences, and
Person 2's attention-mask-aware mean pooling that excludes padding and special tokens. It
prints and writes named geometry metrics. Metrics that were not collected (linear probe,
token geometry, pretrained-BERT analysis, GPU benchmark) appear under `not_run`; never fill
them with invented values. A `status: "degenerate"` result is a valid measured outcome and
must be reported as such.

## 10. Store durable artifacts

Cloud runtime disks are ephemeral. Copy the whole bounded run to Drive or Kaggle output and
record checksums:

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

Do not commit datasets, checkpoints, embeddings or `.venv`. Put the durable location and
the manifest/checkpoint hashes in `docs/PROGRESS.md` only after the run actually completes.

## 11. Required cross-branch checkpoint fix

Person 1's `student.pt` export and training checkpoint store the encoder under `"student"`.
Person 2's current `load_student_checkpoint` selects only `"student_encoder"`, `"model"` or
`"encoder"`, so it does **not** recognise Person 1's payload. `src/evaluate_bridge.py`
handles both shapes and can emit a compatible wrapper:

```bash
python - <<'PY'
from src.evaluate_bridge import load_student_compat, person2_compatible_state

model, info = load_student_compat("student.pt")
print(info["source_kind"])
print(sorted(person2_compatible_state("student.pt")))
PY
```

The durable fix belongs on Person 2's branch: accept the `"student"` key (or the wrapper)
with a schema check and a test. Do not silently edit `src/evaluate.py` from `person1`.

## 12. Evidence checklist

- [ ] GPU runtime selected and `torch.cuda.is_available()` observed true.
- [ ] Exact commit pinned and recorded.
- [ ] Cloud test output recorded (including whether the CUDA test passed).
- [ ] Bounded batch manifests and SHA-256 values recorded.
- [ ] Three pilots completed with measured wall-clock, tokens and losses.
- [ ] Student exports and geometry JSONs copied to durable storage.
- [ ] Person 2 checkpoint-key fix agreed and tested.
- [ ] `docs/PROGRESS.md` updated with real commands and outcomes.

Until all of the above are observed, state that no GPU run has been executed.
