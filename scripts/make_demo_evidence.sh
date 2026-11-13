#!/usr/bin/env bash
# Produce genuine module-wise evidence on CPU from the cached WikiText and AG News datasets.
# Logs land in docs/assets/logs (committed); artifacts stay in scratch.
set -u
cd 'C:/Users/daksh/MyLife/Work/7th-sem'
export OMP_NUM_THREADS=2
A='C:/Users/daksh/AppData/Local/hermes/cache/scratch/demo2'
L='C:/Users/daksh/MyLife/Work/7th-sem/docs/assets/logs'
PY='.venv/Scripts/python.exe'
mkdir -p "$A" "$L"

"$PY" scripts/build_batches.py --dataset Salesforce/wikitext --config-name wikitext-103-raw-v1 \
  --split train --tokenizer bert-base-uncased --max-length 128 --policy span --mask-ratio 0.15 \
  --max-records 600 --batch-size 8 --seed 17 --out "$A/span.pt" > "$L/01_data.txt" 2>&1
echo "build span rc=$?"

"$PY" scripts/build_batches.py --dataset Salesforce/wikitext --config-name wikitext-103-raw-v1 \
  --split train --tokenizer bert-base-uncased --max-length 128 --policy independent --mask-ratio 0.15 \
  --max-records 600 --batch-size 8 --seed 17 --out "$A/independent.pt" > "$L/01b_data_independent.txt" 2>&1
echo "build independent rc=$?"

"$PY" scripts/run_pilot.py --variant latent_a --batches "$A/span.pt" --output "$A/run-a" \
  --max-updates 120 --max-tokens 10000000 --device cpu --seed 17 > "$L/02_train_latent.txt" 2>&1
echo "train A rc=$?"

"$PY" scripts/run_pilot.py --variant mlm --batches "$A/independent.pt" --output "$A/run-c" \
  --max-updates 120 --max-tokens 10000000 --device cpu --seed 17 > "$L/03_train_mlm.txt" 2>&1
echo "train C rc=$?"

"$PY" scripts/run_probe.py --student "$A/run-a/student.pt" --train-limit 2000 --test-limit 500 \
  --batch-size 64 --device cpu --random-baseline --out "$L/04_probe.json" > "$L/04_probe.txt" 2>&1
echo "probe rc=$?"

"$PY" -m src.evaluate_bridge --student "$A/run-a/student.pt" --dataset Salesforce/wikitext \
  --config-name wikitext-103-raw-v1 --split validation --tokenizer bert-base-uncased \
  --max-length 128 --max-records 200 --batch-size 32 --device cpu \
  --out "$L/05_geometry.json" > "$L/05_geometry.txt" 2>&1
echo "geometry rc=$?"

"$PY" -m pytest -q > "$L/06_tests.txt" 2>&1
echo "tests rc=$?"
git log --oneline -3 > "$L/07_git.txt" 2>&1
echo DONE
