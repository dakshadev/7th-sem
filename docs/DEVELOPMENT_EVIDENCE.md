# Development evidence

## 27 October 2026 -- CPU verification follow-up

| phase | actual command | observed outcome |
|---|---|---|
| Environment | `.venv\\Scripts\\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"` | `2.9.1+cpu`, `False`; CPU-only environment. |
| Lint | `.venv\\Scripts\\python.exe -m ruff format --check src tests; .venv\\Scripts\\python.exe -m ruff check src tests` | Passed: 10 files formatted; all checks passed. |
| CPU tests | `OMP_NUM_THREADS=1; MKL_NUM_THREADS=1; .venv\\Scripts\\python.exe -m pytest -q` | **25 passed, 1 skipped** in 13.20s. The CUDA resume test skipped because CUDA is unavailable. |
| CPU C/A/B smoke | `python -m src.synthetic ...; python -m src.train --config configs\\{mlm_c,latent_a,latent_b}.json ...` | All three two-update synthetic diagnostics completed; checkpoints, JSONL metrics and student exports were created under `%TEMP%\\person1-cpu-smoke-20261027`. No cloud/GPU or real-data run occurred. |

The earlier network failures below are historical. The local CUDA prohibition remains in force: do not turn the CPU environment into a CUDA environment or claim cloud results without executing them.

This file records commands actually executed while implementing Person 1. “RED” means a test command was expected to fail because the behavior had not yet been implemented; “GREEN” follows the minimal implementation. Synthetic results here are correctness diagnostics only, not research or transfer results.

## 27 October 2026 — person1

| phase | actual command | observed outcome |
|---|---|---|
| Environment | `uv --version; py -0p; python --version; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader` | `uv 0.12.9`; Python 3.12 and 3.11 were installed; `python` launcher was broken due inaccessible 3.14 path; RTX 4070 Laptop GPU, 8188 MiB, driver 610.60. |
| Environment attempt | `uv venv --python C:\Users\daksh\AppData\Local\Programs\Python\Python312\python.exe .venv` | Initial attempt failed because uv could not access its global cache. |
| Environment GREEN | `$env:UV_CACHE_DIR="$PWD\.uv-cache"; uv venv --python C:\Users\daksh\AppData\Local\Programs\Python\Python312\python.exe .venv` | `.venv` created with CPython 3.12.10. |
| Dependency attempt | `uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch` | Failed after retries: network tunnel connection refused. No CUDA wheel was installed. |
| Dependency attempt | `uv pip install --python .venv\Scripts\python.exe torch pydantic pytest ruff` | Failed after retries: network tunnel connection refused to PyPI. No test/lint dependencies installed. |
| RED config | `C:\Users\daksh\AppData\Local\Programs\Python\Python312\python.exe -c "from src.config import ExperimentConfig"` | Expected failure before implementation: `ModuleNotFoundError: No module named 'src.config'`. |
| GREEN config | `...python.exe -c "from src.config import ExperimentConfig; assert ExperimentConfig().model.width == 256; print('config GREEN')"` | Passed: `config GREEN`. |
| Source syntax GREEN | `...python.exe -m compileall -q src tests` | Passed (no output). |
| Config/file GREEN | `...python.exe -c "from src.config import ExperimentConfig, ModelConfig; import json; assert ExperimentConfig().model.layers == 4; assert ExperimentConfig.from_dict(json.load(open('configs/mlm_c.json'))).objective.variant == 'mlm'; print('config and compile GREEN')"; git diff --check` | Passed: `config and compile GREEN`; no whitespace errors. |
| Config/schema GREEN | `...python.exe -m compileall -q src tests; ...python.exe -c "from src.config import ExperimentConfig, load_config; assert ExperimentConfig().model.layers == 4; assert load_config('configs/latent_a.json').mask.policy == 'span'; assert load_config('configs/latent_b.json').teacher.target_norm == 'layernorm'; print('config schema GREEN')"; git diff --check` | Passed: `config schema GREEN`; no whitespace errors. |
| Final static GREEN | `...python.exe -m compileall -q src tests; ...python.exe -c "from src.config import load_config; assert [load_config('configs/' + x).objective.variant for x in ('mlm_c.json','latent_a.json','latent_b.json')] == ['mlm','latent_a','latent_b']; print('all config variants GREEN')"; git diff --check` | Passed: line-length audit found no Python source/test lines over 100 characters; config variants and whitespace checks passed. |

The required Torch test RED/GREEN cycles, `pytest`, `ruff`, synthetic optimization, checkpoint/export invocation, full-size smoke, and GPU smoke have **not** run because no Torch package could be installed. They remain required before calling the implementation verified.
