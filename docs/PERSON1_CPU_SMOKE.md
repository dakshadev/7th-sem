# Independent Person 1 real-text CPU smoke — 27 October 2026

This is a bounded integration/correctness check, **not** the full pretraining study, a collapse diagnosis, a downstream transfer evaluation, or a GPU result. It ran on CPU only, using the proposed vocabulary/length/width/layer settings. No local CUDA training ran. Free Colab/Kaggle remains the GPU execution target.

The parent executed a separate integration fixture script against the working implementation. Fixture source: canonical WikiText-103 raw **train** and **validation** rows retrieved independently; no cross-split packing. Tokenizer: google-bert/bert-base-uncased at the revision recorded in the attached JSON evidence. The source dataset-server response is not revision-pinned; its response hash is recorded. Only a tiny fixture was used, with a small amount of repeated training exposure. Mask policies differ between A (short spans) and B/C (independent selected positions); all replace selected positions with MASK and exclude special/padding tokens.

## Actual outcomes

| Variant | Updates | Nonpadding training tokens | Masked targets | First train loss | Last train loss | Target-weighted validation loss | Train seconds (CPU) |
|---|---|---|---|---|---|---|---|
| mlm | 20 | 5471 | 801 | 10.448955 | 9.102030 | 9.613659 | 11.229 |
| latent_a | 20 | 5471 | 801 | 1.959867 | 1.322652 | 1.348988 | 9.005 |
| latent_b | 20 | 5471 | 801 | 0.555579 | 0.441495 | 0.428278 | 5.411 |

Loss scales differ by objective; do not compare these numeric values as a model-quality ranking. Falling loss does not establish non-collapse or useful representations. All three checkpoints were reloaded, an additional step executed, and student export outputs compared against the corresponding trained student. Those checks passed. Exact CPU deterministic resume equivalence is tested separately in the unit suite.

## Invocation and artifacts

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv/Scripts/python.exe 'C:/Users/daksh/AppData/Local/hermes/cache/scratch/person1-real-smoke/run_real_smoke.py'
```

- Framework: PyTorch 2.9.1+cpu.
- Local artifacts: `artifacts/person1-realtext-smoke/{mlm,latent_a,latent_b}/` (ignored by Git; contains configs, manifests, batch fixtures, checkpoints, exports and step logs).
- Small evidence manifest: `results/person1/realtext_cpu_smoke.json` (no raw text, tensors or checkpoints).
- Producer script is a parent-side scratch integration tool, not Person 2's production data pipeline. Long-lived reproducible experiments must use the integrated project data pipeline and trainer CLI.
- Execution used an uncommitted development snapshot before final review/publishing; do not attribute these results to a later commit without rerunning. This evidence is limited to training-path functionality. GPU resume, durable cloud recovery, corpus-scale training, geometry on trained outputs and AG News transfer remain separate acceptance items.
