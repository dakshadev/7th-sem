# Project context and scope

## Source and team

Context was extracted from the user's pre-project deck, `Pre_Project_Presentation_TLP_Lab (2).pptx`; the original is not uploaded here. Deck title: **A Lightweight Self-Supervised Framework for Text Representation Learning Using Masked Latent Prediction**. Institution: MSRUAS, Department of CSE; guide: Dr. Prabadevi B; batch PB93. The deck lists three team members, but the user explicitly says **two people are now implementing**. Daksh is Person 1; Person 2's identity is not confirmed. Do not infer the active teammate from the deck.

The implementation state is recorded in `docs/PROGRESS.md`, which is the authoritative log of what ran and what it measured; do not infer progress from this context file. Current planning date: 27 October 2026; mid-review date: 29 October 2026. A previous chat schedule incorrectly started on October 26: do not reuse it. No formal percentage-completion rubric is supplied; review template is not yet in this repository. Expect progress and remaining-work sections, not a completed thesis.

## Research question

Under a fixed small English text encoder, dataset and compute budget, how do teacher targets, mask patterns, predictor design and regression loss influence collapse, stability and useful representations? Explicit anti-collapse regularization is an optional extension.

Hypothesis from the proposal: normalized multi-layer teacher targets may help prevent collapse; lower training loss alone will not reliably indicate better transfer. This is a hypothesis, not a result. One-factor ablations measure effects around a chosen recipe; they do not establish a universally decisive component or resolve component interactions.

## Proposed objectives

1. Profile corpus token frequencies, lengths and masked-word ambiguity.
2. Implement EMA-teacher masked latent learning and an MLM control with the same small encoder/data.
3. Test normalized multi-layer targets; optional SIGReg if time permits.
4. Compare component switches using geometry, frozen transfer and measured compute.

## Fixed architecture/data starting point

- WikiText-103 subset: proposal target about 5 million tokens. Start with a smaller explicitly labelled pilot if needed, without silently changing the final target.
- `bert-base-uncased` tokenizer, proposed vocabulary 30,522. Verify actual tokenizer metadata and record its revision.
- Sequence length 128; width 256; 4 encoder layers; 4 attention heads.
- Learned token and position embeddings; all encoder weights random at initialization.
- Additional settings are now fixed in `src/config.py`, which is the source of truth: FFN width 1024, dropout 0.1, pre-norm, EMA decay 0.996 applied after each optimiser step, AdamW with linear warmup and decay, batch size 8, fp32. Change them there, not in prose, and keep them shared across comparisons where applicable.

## Proposed variants

| ID | Teacher target | Mask | Predictor | Loss |
|---|---|---|---|---|
| A | Final teacher layer | Short contiguous spans, about 15% eligible tokens | 2-layer attention predictor | MSE |
| B | Mean of normalized top-2 teacher layers | Independent tokens, about 15% eligible tokens | Linear projection | Smooth L1 |
| C | True token identities (no teacher) | Independent tokens | Vocabulary classifier | Cross-entropy |
| D | A with only B's target construction | As A | As A | As A |
| E | A with only B's masking | As B | As A | As A |
| F | A with only B's predictor | As A | As B | As A |
| G | A with only B's loss | As A | As A | As B |
| H (optional) | As A | As A | As A | MSE plus SIGReg |

A and B deliberately change multiple components and only compare recipes; D-G provide local single-change comparisons. A linear-predictor smoke configuration must not be called A. Normalization axes/type, span algorithm and predictor details must be made explicit before comparing runs.

## Review scope and remaining research

Immediate scope is met: reproducible data preparation, tested shared encoder, MLM and latent runs, checkpoint recovery, loss/geometry plots, A/B/C pilots and the AG News frozen probe all exist. Remaining work: D-G ablations are implemented and run at one seed only, so they need probes and a second seed; C/A/B already have two seeds; the 0.69M vs 5M budget sweep is done. What remains after that is optional SIGReg and the second probe task, SST-2.

The proposal originally described 16 weeks and a fallback 20-40 GPU-hour estimate. Neither is proof of work completed or an available GPU allocation. Free Colab/Kaggle quota and throughput must be measured. Do not claim full literature novelty or paper findings have been independently reverified here; the deck's literature summary requires source checking before thesis-level claims.
