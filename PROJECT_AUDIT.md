# Independent audit — masked latent prediction project

**Audited tree:** `main` @ `137d114`, clean, level with `origin/main`.
**Audit date:** 25 November 2026.
**Scope:** the repository as checked out — source, configs, tests, result JSON, documentation.
**Method:** every figure below was re-derived by script from `results/person1/*.json` or
executed (`python3 -m pytest`, a parameter-count pass over the real modules). Nothing is quoted
from a document without checking the artifact it cites.

**Implementation status (25 November 2026):** the P0-P3 code items in §4 are implemented on
`audit/project-review`; the experiments they enable have not been run. `docs/PROGRESS.md`
records each interval with its command and outcome, and `docs/EVIDENCE_TABLES.md` carries the
parameter and measured-cost tables.

**Audit of that branch (added after the implementations landed).** Reviewing the implementation
found six defects, now fixed and recorded under "Audit of the audit branch" in
`docs/PROGRESS.md`. Two were regressions that would have blocked work: every checkpoint written
on `main` refused to resume (the `[UNK]` eligibility fix changed `special_token_ids`, and the
resume check compared whole configs), and `scripts/run_sweep.py` aborted whenever a budget
derived fewer updates than the configured warm-up. Three were provenance or correctness
problems: `configs/latent_a.json` had been altered so it no longer identified the runs in
`results/`, the notebook cloned a branch without the runner it calls, and the evidence tables
printed C's head size for M. One was a wrong number: the transfer-margin range in
`docs/DATA_INTEGRITY.md`. Findings 3.1.1-3.1.5, 3.2 and 3.3.3 in the body of this document
remain accurate as descriptions of `main`; the fixes for them are visible in the branch diff.

**What was not inspected:** the Kaggle run trees, the Kaggle kernel source for
`dakshasubramanya/7th-sem-budget-sweep` (v3) and `7th-sem-probe-only`, the `.pptx` / `.docx`
artefacts, and `results/Multi_doc_compare/` (an empty directory). Several claims in
`docs/DATA_INTEGRITY.md` can only be checked against the Kaggle tree; where that is the case the
verdict below says so rather than assuming either way.

---

## 1. What the project is, and why

The research question, from `README.md` and `docs/PROJECT_CONTEXT.md`: under a fixed, small,
random-initialised English text encoder with a fixed dataset and compute budget, does a **masked
latent** objective — predicting a teacher's own hidden states at masked positions — produce
better representations than **masked token** prediction, and which component of the recipe
causes any difference?

That is a good question and it is stated narrowly enough to be answerable. The project's own
framing in `README.md` is the sharper version: masked-token prediction spends capacity on lexical
identity, so a latent target promises semantic targets instead — *if* the target does not
collapse to a constant.

### Fixed design

Verified against `src/config.py`, `configs/*.json`, `src/model.py`, `src/objectives.py`:

| Element | Value | Source |
|---|---|---|
| Encoder | 4 layers, width 256, 4 heads, FFN 1024, seq 128, pre-norm, dropout 0.1 | `src/config.py:36-46` |
| Embeddings | learned token + learned absolute position, all weights random at init | `src/model.py:29-33` |
| Parameters | **11,005,952** — 7,813,632 token emb, 32,768 position emb, 512 norm, 4 × 789,760 layers | measured this session |
| Tokenizer | `bert-base-uncased`, tokenizer only; no pretrained encoder weights anywhere | `src/data.py:33-39` |
| Teacher | EMA copy of the student, decay 0.996, targets detached, `eval()` | `src/train.py:312-322`, `src/objectives.py:48-68` |
| Masking | ratio 0.15 of eligible positions; selected positions replaced by `[MASK]` | `src/data.py:108`, `src/data.py:202` |
| Conditions | C (MLM), A, B (latent anchors), D–G (one-change ablations of A), H/I (anti-collapse auxiliaries), J (no teacher) | `configs/`, `README.md:23-31` |

The `[MASK]`-only corruption is correct per the project's own contract, not an oversight:
`docs/INTERFACES.md` explicitly says "Do not silently assume BERT 80/10/10 corruption". It is
still a substantive design choice that has never been ablated (finding 8).

### Evidence inventory

Every file in `results/person1/`, what produced it, and what it is allowed to support. This
table is the constraint the rest of the document works inside.

| File | Commit | Contents | May support | May **not** support |
|---|---|---|---|---|
| `kaggle_t4_pilot.json` | `fbc93db` | 3 runs × 50 updates, 22,833 tokens, 3,333 masked targets, 400 WikiText-2 records | the pipeline runs on a real T4 and writes comparable artefacts | any design conclusion (explicit `status: bounded_pilot_only`) |
| `kaggle_long_runs.json` | `ee8676b` | 6 anchor runs (C/A/B × 2 seeds) at 700,482 tokens / 1,230 updates; 4 ablations D–G at seed 17; 3 short runs | loss trajectories at 0.7M tokens; that D–G ran | anything about transfer; D–G comparisons at all (loss only, 1 seed, different scales) |
| `geometry_long.json` | `ee8676b` | effective rank for 6 students, 300 WikiText-2 validation records | no condition is degenerate at that budget; MLM's space is much lower-rank | that the *targets* are healthy; that rank predicts transfer |
| `probe_agnews_long.json` | `ee8676b` | 6 frozen AG News probes + 1 random-init reference, per-student SHA-256 | every probed student is below the random reference on this task | that the objective is worse than random init (n=1 baseline, lexical task) |
| `probe_agnews_t4.json` | `c0f330a` (in `_provenance`) | 3 pilots probed + the same reference | same | same |
| `ambiguity_wikitext2.json` | `ee8676b` | BERT-token entropy over 26,840 positions, 400 records | masked text is not a deterministic target | that a latent target is better posed than a token one |
| `sweep_annealed.json` | `5f20e6a` | **12 runs**: C/A/B × 2 seeds × {0.69M, 5M}, annealed LR | the headline: transfer falls with budget and stays below the reference | that D–G components matter; that the anneal was reproducible from this repo (it was not — §3.2) |
| `sweep_anticollapse.json` | `2a54edc` | **12 runs**: H/I/J × 2 seeds × {0.69M, 5M}, with probe + geometry | the guards do not fix transfer; removing the teacher helps slightly | that A's 5M rank exists (it is missing — see below) |
| `sweep_pass1_no_annealing.json` | `5f20e6a` | superseded 12 runs, `max_updates=40000` | nothing; retained for the record | **must not be cited as the budget comparison** |
| `realtext_cpu_smoke.json` | — | a tiny real-text CPU fixture with tokenizer/dataset revisions | the tokenizer and dataset revisions actually used | any training result |

One gap is reported honestly in `README.md:127-129` and is worth repeating: **A's 5M effective
rank does not exist.** The annealed sweep's geometry output was not retrieved from Kaggle, so
`sweep_anticollapse.json` has rank for H, I, J at 5M but not for A, and `geometry_long.json`
covers only the 0.7M students. The README says so instead of guessing. That is the right
behaviour and it should not regress into an estimate later.

---

## 2. What the evidence actually says — claim-by-claim

| Claim | Verdict | Basis |
|---|---|---|
| The pipeline is real, tested and reproducible run-to-run | **Supported** | 64 test functions / 66 collected items, all pass; checkpoint-and-RNG parity test; batch manifests with SHA-256; per-run metadata with commit, device and source fingerprint |
| Ablations D–G exist as one-change configs | **Supported, with a caveat** | D changes `objective.target` **and** the coupled `teacher.target_norm`; F and G drop the now-inert `predictor_dropout`. Semantically one change each, textually two fields |
| D–G tell us which component matters | **Not supported** | 700k tokens, one seed, loss only, and the losses are on different scales — F ends at 0.9015 against A's 0.5052 because Smooth L1/MSE have different ranges, not because F is worse |
| No trained student beats random init | **Partly** | True in **all 33 probed students** (highest anywhere 0.607 vs 0.632), but the reference is a single unpinned draw — n = 1 |
| More budget hurts transfer | **Supported as measured, and confounded** | 12 of 12 (condition, seed) cells degrade, by 0.013–0.071. But EMA horizon and LR schedule both change with budget (§3.3, item 9) |
| Nothing collapses | **Supported for the student only** | geometry is computed on student outputs over clean text; the latent *targets* are never measured |
| Masked text is ambiguous enough to motivate latents | **Partly** | shows token-level ambiguity (1.277 nats mean), does not test whether a latent target is better posed than a token target |

### The two headline findings, restated precisely

**Finding 1 — no trained student beats a random-init encoder on this probe.** 33 students across
four files, best 0.607 (`sweep_anticollapse.json`, I at seed 43, 0.69M), reference 0.632. The
margin in the annealed sweep ranges −0.037 to −0.108.

**Finding 2 — more pretraining makes frozen transfer worse.** Means over the two seeds:

| Condition | 0.69M | 5M | Drop |
|---|---|---|---|
| C — MLM | 0.5765 | 0.5315 | 0.045 |
| A — latent | 0.5895 | 0.5490 | 0.0405 |
| B — latent | 0.5860 | 0.5370 | 0.049 |

Per-seed drops range 0.027 (B, seed 43) to 0.071 (B, seed 17). In the anti-collapse sweep
H/I/J drop by 0.013–0.047. **Every measured cell degrades.** That consistency is the strongest
thing in the evidence base, and it is what makes the result worth explaining rather than
dismissing.

Note that `README.md:113-114` and `docs/PROGRESS.md:134` summarise this as "0.04–0.05". That is
true of the three seed-mean drops but understates the spread: individual cells move by as little
as 0.027 and as much as 0.071. Since the documented probe noise floor is ~0.002, all of these
are real, but they are not all the same size.

---

## 3. What is wrong

Ordered by severity. Each item gives the evidence, why it matters, and what it costs to fix.

### 3.1 Evidence-integrity defects — highest severity, zero compute to fix

These are the most serious findings in this audit because they are the class of error
`AGENTS.md` rule 1 forbids: a claim presented as measured that the measured artefact
contradicts.

**3.1.1 `docs/DATA_INTEGRITY.md` misreports the stopping state of the file it audits.**

`docs/DATA_INTEGRITY.md:20-21` states, under "Verified":

> `stop_reason == "budget"` in all 12, with token counts within one batch of the target
> (700,482 vs 700,000; 5,000,623 vs 5,000,000).

The file it audits is `results/person1/sweep_annealed.json`. That file says otherwise for half
its runs:

- All six 0.69M runs have `"stop_reason": "interrupted"` and `"nonpadding_tokens": 692919`.
- 692,919 is **7,081 tokens short** of 700,000 — about 12.4 batches at the measured 569.4
  tokens/update, not "within one batch".
- 700,482 appears in a *different* pass: `kaggle_long_runs.json` (`"per_run_nonpadding_tokens":
  700482`, 1,230 updates, commit `ee8676b`). It is not this file's number.

`docs/PROGRESS.md:132` repeats the same false claim in the budget-sweep entry: "0.69M runs stop
at 692,919 tokens / 1,217 updates; 5M runs at 5,000,623 tokens / 8,668 updates; all stop_reason
`budget`." The first half of that sentence contradicts the second half.

*Why it matters:* the 0.69M tier is one half of the project's headline comparison. If it stopped
on a step cap rather than the token budget, the "0.69M vs 5M" framing is really "1,217 updates
vs 8,668 updates with a truncated token budget at the low tier". The conclusion probably
survives — the 5M tier genuinely hits its budget, and 7,081 tokens is 1% — but a reader cannot
know that from the audit, because the audit asserts the thing that is false. Any examiner who
opens the JSON finds the discrepancy immediately.

*Cost to fix:* minutes. Correct the claim, state the 7,081-token shortfall and its cause, and
keep the 5M tier's `budget` stop as the evidenced one.

**3.1.2 `src/evaluate_bridge.py` ships a false compatibility note, and it propagates.**

`COMPATIBILITY_NOTE` at `src/evaluate_bridge.py:32-36` says Person 2's loader "does not recognise
Person 1's `'student'` key". `src/evaluate.py:277` in this same tree reads:

```python
if "student" in state:
    weights = state["student"]
```

`"student"` is checked **first**, before `student_encoder`, `model` or `encoder`
(`src/evaluate.py:277-286`). The note was true before the `2273c01` fix and was never updated —
`docs/PERSON2_REVIEW.md` records the fix and `docs/PROGRESS.md:108` says the loader now accepts
`student`.

The note is not inert. `evaluate_student` puts it in the `limitations` list of every geometry
report it writes (`src/evaluate_bridge.py:329`). It appears verbatim in
`docs/assets/logs/05_geometry.json`. So the project publishes, as a limitation of a measured
result, a defect that does not exist.

*Cost to fix:* one string. Delete it, and regenerate or annotate the geometry log that carries it.

**3.1.3 Stale dates in `docs/PROJECT_CONTEXT.md`.**

`docs/PROJECT_CONTEXT.md:7` says "Current planning date: 27 October 2026; mid-review date: 29
October 2026". Today is 25 November 2026 — the mid-review is 27 days in the past and is not
mentioned anywhere else in the file. This is a small thing that signals the context document is
not maintained, which matters because `AGENTS.md` tells every assistant to read it first.

**3.1.4 `AGENTS.md` contradicts `README.md` on the test suite.**

`AGENTS.md:10` says "the suite is `55 passed, 2 skipped`". `README.md:54` and `README.md:218` say
"64 passed, 2 skipped". Executed this session:

```
64 passed, 2 skipped in 21.22s       (66 items collected)
```

`AGENTS.md` is stale. Because it is the first file any assistant reads, it is the highest-traffic
wrong statement in the repository.

**3.1.5 `docs/PROGRESS.md` claims a `requirements.txt` warning that is not there.**

`docs/PROGRESS.md:110` records, among the Person 2 fixes: "`requirements.txt`: warns that cloud
runtimes must not install torch from it." The file contains eight dependency lines and **no
comments at all** (`requirements.txt:1-8`, `torch>=2.0.0` on line 1). `git log -- requirements.txt`
shows one commit, `8ee1a18`, adding it as-is.

This is the same defect class as 3.1.1: a recorded fix that was not applied. It matters more than
it looks, because the underlying hazard is real and was actually hit — `docs/PERSON2_REVIEW.md`
defect 2 documents `pip install -r requirements.txt` replacing CUDA torch with `2.11.0+cpu` on
Kaggle (see 3.6). The mitigation exists only in the notebook and in prose; the file that people
will actually install from still carries the trap.

### 3.2 Reproducibility: the producer of the headline numbers is not in the repository

`sweep_annealed.json:10` states:

> max_updates is derived from the budget (budget/575) so the cosine-free linear schedule reaches
> zero at the end of every run.

I tested that claim against the tree. It is internally consistent — `700000/575 = 1217.39`, and
the JSON records exactly 1,217 updates; `_schedule_scale` (`src/train.py:155-159`) returns
`(maximum - update) / (maximum - warmup)`, which at `update == max_updates == 1217` is exactly
0.0, i.e. annealed to zero as described. The six 0.69M runs also carry `interrupted`, which is
what `run_batches` writes when a segment cap fires (`src/train.py:509-514`) — consistent with a
harness that passed `segment_updates = max_updates = 1217`.

What the repository cannot do is reproduce any of it:

- **No config carries an `optimizer` block.** All ten files in `configs/` omit it (`grep -l
  '"optimizer"' configs/*.json` returns nothing), so `OptimizerConfig.max_updates` falls back to
  its default of `10_000` (`src/config.py:83`). At 1,217 updates the schedule would sit at
  `(10000-1217)/(10000-100) = 0.8872` — 89% of peak LR, not zero.
- **No script derives `max_updates` from the budget.** `grep -rn 575` across the tree returns
  only documentation and deck prose (`docs/PROGRESS.md:131`, `docs/assets/deck_payload.json:89`,
  `scripts/extend_deck_for_review.py:64-65`, `scripts/build_synopsis.py:387`) and the JSON's own
  note. No Python file computes it.
- **No sweep script was ever committed.** `git log --all --oneline -- '*sweep*'` returns two
  commits, and both only add result JSON.
- **`notebooks/` holds only `person1_cloud_pilot.ipynb`**, a 7-cell bounded pilot whose cell 3
  builds a 400-record WikiText-2 artefact — not the 90,000-record budget sweep.

`docs/PROGRESS.md:130` names the actual producer: the Kaggle kernel
`dakshasubramanya/7th-sem-budget-sweep` (v3), with a second kernel `7th-sem-probe-only` for the
re-probe. Neither is in the repository.

*Consequence:* the checked-in tree cannot regenerate `sweep_annealed.json`,
`sweep_anticollapse.json`, the probes, or the geometry. Only the JSON survives. The provenance
chain for the project's central result terminates in a Kaggle session that no longer exists.

*Cost to fix:* a day, not a GPU-hour. Add `scripts/run_sweep.py` that derives `max_updates` from
the observed tokens-per-batch of a manifested artefact, records the derivation in the run
metadata, and export the kernel source into `notebooks/`. Nothing needs re-running to close this
— the script must reproduce the recorded *configuration*, and its correctness can be checked by
showing it derives 1,217 from a 0.69M budget.

### 3.3 Training-path defects that plausibly cause the negative result

**3.3.1 There is no shuffling anywhere on the pilot path.**

- `scripts/build_batches.py:196` chunks records with `for text in texts:` in corpus order.
- Batches are consecutive slices: `records[start:start+batch_size]` (`scripts/build_batches.py:217-218`).
- `StatefulBatchSequence` (`src/train.py:83-99`) walks that list by index, forever, from position 0.

So every batch in every run is 8 consecutive WikiText lines — usually the same article — and
every run sees the corpus in the same order. `create_dataloader(..., shuffle=True)` exists
(`src/data.py:371`, and `src/data.py:519` passes `shuffle=True`) but is not on the path that
produced any result: `scripts/run_pilot.py` builds a static artifact and hands it to
`StatefulBatchSequence`.

*Why it matters:* this is the strongest candidate for a mechanical cause of the negative result
that has nothing to do with the objective. Consecutive-line batching gives extremely correlated
gradients within a batch and a global curriculum in document order; the model can fit the
first N articles and never revisit them. It makes the two seeds far less independent than they
look, and it interacts with the finding that more budget hurts: more budget means visiting more
distinct articles, which for a 256-wide 4-layer model may be more forgetting than learning.

**3.3.2 Masks are baked into the artefact and never resampled.**

`build_batches` calls `process_single_text` once per record (`scripts/build_batches.py:200-208`)
and `torch.save`s the finished tensors (`scripts/build_batches.py:325`). The trainer reads
`masked_input_ids` and `prediction_mask` as given; nothing in `src/train.py` re-masks.

*Why it matters:* masking is a stochastic augmentation, and the design intends it to be a free
variable. Because the artefact is per-corpus and the harness reuses it, the same mask
realisation is seen at every epoch, and — per `README.md:117`, which says the anti-collapse sweep
used "same batch artifacts as the sweep above" — seeds 17 and 43 share one mask realisation *and*
one batch order. The two seeds therefore bound initialisation variance only, not data-order or
masking variance. Every "two seeds" claim in the project inherits that limitation. It is not
stated anywhere.

**3.3.3 `mask.span_length` is dead configuration.**

`MaskConfig.span_length` is declared and validated (`src/config.py:136`, `src/config.py:143`) and
set to 3 in every config. It never reaches the masker: `make_batches` calls `process_single_text`
without `mean_span_length` or `max_span_length` (`scripts/build_batches.py:200-208`), and
`run_pilot._build_namespace` (`scripts/run_pilot.py:86-102`) does not forward them either. So
`src/data.py`'s defaults apply — `DEFAULT_MEAN_SPAN_LENGTH = 3`,
`DEFAULT_MAX_SPAN_LENGTH = 5` (`src/data.py:22-23`).

The values happen to agree, so **no result is wrong**. The configuration is a lie about control:
a user setting `span_length: 7` would get spans of geometric mean 3, capped at 5, and no
warning. This is a latent trap for the next experimenter, and ablations E and D are exactly the
kind of experiment that will try to vary it.

**3.3.4 Corruption is `[MASK]`-only, 100% of selected positions.**

`src/data.py:108` and `src/data.py:202`: `masked_input_ids[prediction_mask] = mask_token_id`. No
random-token or keep-original branch, and no configuration to enable one.

This is deliberate and contract-compliant (`docs/INTERFACES.md` forbids silently assuming
80/10/10), so it is not a defect in the sense of a bug. But it is the single most obvious
untried variant, and BERT's 80/10/10 exists precisely to blunt the train/evaluation mismatch that
`README.md:142-145` names as the leading explanation for the negative result. If the mismatch
hypothesis is right, a corruption policy that sometimes shows the student an unmasked token
should partly close it. That experiment has never been run, and it is cheap.

**3.3.5 The EMA teacher's horizon changes by 7× across the budget comparison.**

Decay is fixed at 0.996 for every run (`configs/*.json`, `src/config.py:114`). The mean lag of an
EMA with decay *d* is *d*/(1−*d*) ≈ **249 updates**; the half-life is ≈173 updates.

| Run length | EMA window as a fraction of the run |
|---|---|
| 1,217 updates (0.69M) | 20.5% |
| 8,668 updates (5M) | 2.9% |

*Why it matters:* the budget comparison is described as varying budget. It also varies the
teacher. At 0.69M the teacher is a substantially lagged, much-smoother model relative to the run
length; at 5M it tracks the student closely. So A/B's teacher is a different object at the two
budgets, and "more budget hurts" mixes more data with a 7× shorter effective target horizon.
C has no teacher, so its 0.045 drop is the cleanest of the three, but C is also the condition
with a 7.9M-parameter head.

### 3.4 Evaluation defects

**3.4.1 The random baseline is a single draw.**

`scripts/run_probe.py:147-186` (`random_baseline`) constructs one `SmallEncoder`, seeds once with
`torch.manual_seed(seed)` (`:161`), and probes it. There is one reference number for the whole
project.

Two facts make this worse than "n = 1". First, both probe files record **identical** baseline
figures — `accuracy 0.632`, `macro_f1 0.6280428897446606` — which is expected given the same
seed, but means the project has one baseline model, not two independent ones. Second,
`docs/DATA_INTEGRITY.md:62-67` already concedes the reference is not hash-pinned
(`student_sha256: null`) and reports a spread of "0.632 in one probe file, 0.631 in another".
Checked: **both probe files say 0.632.** The 0.631 is in `sweep_pass1_no_annealing.json:7`, a
sweep file — so the audit's own note about its own files is misattributed. The substantive point
stands and is slightly stronger than stated: the observed baseline range is 0.631–0.632 across
the project, from draws that are not saved.

*Why it matters:* "no trained student ever beats random init" is the project's most-quoted
result. With n = 1 it is a comparison against one arbitrary point. Random-init encoders vary —
the spread in the fine-tuning literature is not negligible, and nothing here rules out that the
reference happens to be a lucky draw. That is not a reason to disbelieve the finding; it is a
reason the finding is currently unfalsifiable in the direction that matters.

*Cost to fix:* near zero. Ten seeds, mean ± sd, one CPU job.

**3.4.2 The probe read-out is narrow.**

- Pooling excludes `[CLS]` and every special token (`src/evaluate.py:24-64`) — so the representation
  the probe sees is a mean over content tokens only.
- Only the **final** layer is probed: `scripts/run_probe.py:92` takes
  `output.last_hidden_state` and nothing else, even though
  `SmallEncoder(..., return_hidden_states=True)` returns all four (`src/model.py:62-76`).
- The head is a single `LogisticRegression(C=1.0, max_iter=1000, lbfgs)` with a `StandardScaler`,
  no validation split, no tuning (`src/evaluate.py:334-342`, `src/evaluate.py:361-378`).

**3.4.3 The task is lexical, and the baseline is essentially bag-of-words.**

AG News is 4-class topic classification; most of its signal is the presence of topic words. A
random 4-layer encoder mean-pooled over content tokens produces something close to a linear
function of its token embeddings — and its token embeddings are random but *fixed*, which is
exactly a random projection of a bag-of-words. So 0.632 is approximately a bag-of-words
classifier on an easy task.

There is no TF-IDF + logistic baseline anywhere in the repository (`grep -rni 'tfidf|tf-idf|
countvector|bag.of.words'` over `*.py` and `*.md` returns nothing). That baseline is the real
lexical floor on AG News and would very likely land near 0.9. Its absence is why the discussion
cannot currently distinguish "the objective destroyed useful features" from "the probe cannot
detect useful features on this task". This is the most likely reason random is hard to beat, and
it is a statement about the probe at least as much as about the objectives.

**3.4.4 Probe splits are unstratified and taken from the head of each split.**

`scripts/run_probe.py:46`: `load_dataset(dataset, split=f"{split}[:{limit}]")`. That is the first
4,000 training rows and the first 1,000 test rows, in dataset order. Resulting class counts,
recorded in the JSON itself:

- train (4,000): 998 / 797 / 988 / 1,217 — a 1.53× imbalance between the smallest and largest class.
- test (1,000): 268 / 274 / 205 / 253.

Both sets are unstratified and neither is shuffled. A contiguous head-of-split sample may also
correlate with source or date ordering. The majority baseline of 0.274 is computed on the test
set and is fine, but the probe's train/test composition is an accident of dataset order.

**3.4.5 There is no teacher-target geometry.**

`geometry_long.json` is computed by `src/evaluate_bridge.py:evaluate_geometry`, which embeds
**clean, uncorrupted** text with the student and pools it (`evaluate_bridge.py:172-201`). The
`evaluation.text_corruption` field records `"none"`.

So "nothing collapsed" is a claim about student outputs on clean text. Nothing in the repository
measures the rank, variance or cosine structure of the *latent targets* — the quantities that
would actually tell you whether the teacher is providing a rich or a degenerate signal. The
project's core motivation is "a latent target is only useful if it does not collapse to a
constant", and the one measurement that would test it is missing. `rank` of the student's output
is not a proxy: a constant target can produce a low-rank student, and a full-rank target can
produce a collapsed one.

**3.4.6 Objective 1 does not test the project's premise.**

`ambiguity_wikitext2.json` measures the predictive entropy of `bert-base-uncased` at one masked
position at a time. It establishes that masked text has several plausible fillers (mean 1.277
nats, median 0.602, 30.2% of positions above log 5). That is a real and correctly-measured fact.

It does not establish the premise the project needs, which is that a *latent* target is better
posed than a *token* target. Token entropy shows the token target is ambiguous; it says nothing
about whether the latent target is more or less ambiguous, more or less linearly readable, or
more or less useful. The inferential gap from "the token target is ambiguous" to "therefore
predict the latent" is exactly where the project's hypothesis lives, and it is unbridged.

The same analysis also **removes** a motivation rather than supplying one: span masking is not
more ambiguous than independent masking (1.633 vs 1.612 nats; 191 of 400 records span-more-
ambiguous, 202 independent-more-ambiguous). The earlier 12-record result that motivated span
masking did not survive at n = 400. That is reported honestly in `README.md:178-180`.

### 3.5 Reporting asymmetries

**3.5.1 Trainable parameters differ by up to 1.7× and this is never tabulated next to accuracy.**

Measured this session from the real modules:

| Condition | Encoder | Extra head | Trainable total | Second encoder forward? |
|---|---|---|---|---|
| C — MLM | 11,005,952 | 7,910,458 (`MLMHead`) | **18,916,410** | no |
| A — latent | 11,005,952 | 1,678,080 (`AttentionPredictor`) | **12,684,032** | **yes**, per step |
| B — latent | 11,005,952 | 65,792 (`LinearPredictor`) | **11,071,744** | **yes**, per step |

The MLM head is 7.91M parameters — of which the output decoder alone is 7,844,154, almost
exactly the size of the token embedding, because it is a `width → 30522` projection. C therefore
carries **6.23M more trainable parameters** than A, a 1.49× total and a 4.71× ratio on the head.

Two asymmetries run in opposite directions and neither is quantified:
- C's head consumes far more of the optimisation budget per update; A and B run a **second full
  encoder forward pass** per step for the teacher (under `no_grad`, `src/train.py:203-206`), which
  C does not.

`docs/PERSON1_LONG_PILOT.md:44-46` names exactly this ("Whether that reflects the objective or
the fact that MLM's head consumes a much larger fraction of the update budget is not
established here") and stops there. The `seconds` field exists in every run of both sweep JSONs
and has never been tabulated beside accuracy.

From the JSON, for completeness — this table does not exist anywhere in the repo:

| Condition | Budget | Seconds (mean of 2) | Tokens/s | Accuracy (mean) |
|---|---|---|---|---|
| A | 0.69M | 45.1 | 15,364 | 0.5895 |
| A | 5M | 316.0 | 15,825 | 0.5490 |
| B | 0.69M | 35.3 | 19,602 | 0.5860 |
| B | 5M | 246.2 | 20,311 | 0.5370 |
| C | 0.69M | 49.3 | 14,061 | 0.5765 |
| C | 5M | 352.8 | 14,176 | 0.5315 |

Note what it shows: **A costs 28% more wall time than B for a 0.004 accuracy advantage at 0.69M
and 0.012 at 5M.** The attention predictor's 25.5× parameter count over B's linear projection buys
about a third of a point at the lower budget and just over a point at the higher one — well
inside what two seeds can resolve. C is the most expensive per token and the lowest-scoring
condition at 0.69M, which is consistent with, but not proof of, the "head eats the budget"
hypothesis.

### 3.6 Environment

**3.6.1 `requirements.txt` pins an unpinned torch and carries no warning.**

`requirements.txt:1` is `torch>=2.0.0`. `pyproject.toml` pins `torch==2.9.1`, so the two files
disagree, and `requirements.txt` is the one a new contributor is likely to run.

`docs/PERSON2_REVIEW.md` defect 2 records what happened when it was: on a Kaggle GPU runtime the
install replaced the image's CUDA build, and the measured result was
`TORCH 2.11.0+cpu  CUDA False  device cpu`. That is a silent CPU fallback — a run that looks like
a GPU run and is not.

Every result in `results/person1/` records `torch: 2.11.0+cu128` and `cuda_available: true`, so
the *measured* runs are unaffected. The hazard is to the next run. `docs/PROGRESS.md:110` claims
the file now warns; it does not (3.1.5).

---

## 4. What to correct — prioritised

### P0 — no compute required

1. Fix `docs/DATA_INTEGRITY.md:20-21` and `docs/PROGRESS.md:132`: the six 0.69M runs stop
   `interrupted` at 692,919 tokens, 7,081 short; only the 5M tier stops on `budget` (3.1.1).
2. Delete or correct `COMPATIBILITY_NOTE` (`src/evaluate_bridge.py:32-36`) and the geometry log
   that quotes it (3.1.2).
3. Update `docs/PROJECT_CONTEXT.md:7` dates (3.1.3).
4. Update `AGENTS.md:10` to 64 passed (3.1.4).
5. Add the warning to `requirements.txt` that `docs/PROGRESS.md:110` already claims is there, and
   reconcile it with `pyproject.toml`'s `torch==2.9.1` (3.1.5, 3.6.1).
6. Add `scripts/run_sweep.py` + the kernel source under `notebooks/` so the headline configuration
   is derivable from the tree (§3.2).
7. Wire `mask.span_length` through `make_batches`, or delete it from `MaskConfig` — do not leave a
   validated setting that does nothing (3.3.3).
8. Correct `docs/DATA_INTEGRITY.md:66` to say where 0.631 actually lives (3.4.1).

### P1 — cheap: CPU, or one short GPU job

9. Random baseline × 10 seeds; report mean ± sd as the reference band, and save the weights or at
   least the seed list (3.4.1).
10. TF-IDF + logistic on the identical splits. Without this number the probe result cannot be
    interpreted (3.4.3).
11. Stratified, shuffled probe splits; record the sampling procedure in the probe JSON (3.4.4).
12. Per-layer (0–4) × {mean-pool, CLS} probe matrix. `return_hidden_states` already exists; this is
    one loop (3.4.2).
13. Tabulate parameters and seconds-per-token beside accuracy for every condition (3.5.1).

### P2 — training path

14. Shuffle records at artefact-build time and/or batch order per pass (3.3.1).
15. Move masking into the training loop so it resamples each epoch; then seeds vary data as well
    as initialisation (3.3.2).
16. Add an 80/10/10 corruption option and make it a configured, recorded policy (3.3.4).
17. Replace fixed decay 0.996 with a value chosen from run length (0.996 → 0.9999) and record the
    effective horizon in the run metadata (3.3.5).

### P3 — the controls that decide the interpretation

18. **Fixed / random-target control.** Same encoder, optimiser, data and budget; replace the
    teacher targets with a constant, or with noise matched in scale. If the probe falls just as
    far, the degradation is optimisation-induced and has nothing to do with the latent objective.
    This is the single most informative cheap experiment in the whole list, because it is the one
    that can make the headline finding *uninterpretable as evidence about objectives* — which is
    exactly what a good control is for.
19. **Head-capacity control.** MLM with a reduced or factorised head, so C and A carry comparable
    trainable parameters. Currently C holds a 6.23M-parameter advantage and A holds a
    second-forward-pass disadvantage; neither is controlled (3.5.1).
20. **Teacher-target geometry.** Measure rank, per-dimension variance and pairwise cosine of the
    latent targets themselves, at masked positions, per condition and budget. This is the
    measurement that tests the project's actual premise (3.4.5).
21. **Layer probe.** If intermediate layers hold their probe accuracy while layer 4 drops, the
    result is a lexical→contextual shift, not damage. That re-frames the entire conclusion from
    "pretraining destroyed the features" to "pretraining moved them out of the final layer" —
    which is a much more interesting and much more defensible finding.

---

## 5. What to do next to give the idea a strong base

### Spend the budget

The recorded costs, from the `seconds` field of `sweep_annealed.json`:

| Run | 5M tokens | 50M tokens (×10, linear) |
|---|---|---|
| A | 316 s | ~0.88 h |
| B | 246 s | ~0.68 h |
| C | 353 s | ~0.98 h |

The entire 12-run annealed sweep logged **0.58 GPU-hours** of training time (sum of `seconds`,
2,089.4 s); `docs/PROGRESS.md:130` records ~50 minutes wall clock for the whole kernel including
probes. Kaggle's free tier is 30 GPU-hours/week.

A 50M-token version of the same 12-run grid is therefore **≈5.1 GPU-hours** — about 17% of one
week's free quota, or under two hours on the two-T4 kernel that produced the current data. Ten
times the budget for a fifth of a week.

This is the highest-leverage move available and it is affordable. The project itself says the
bottleneck is budget (`docs/PERSON1_LONG_PILOT.md:76-79`: 700k tokens is "far below the regime
where the literature's claims were established"). If that is true, the current negative result is
a measurement of the wrong regime; if it is false, the current result is much more interesting
than the project thinks. Either way, spending the budget is what resolves it — and at 5 GPU-hours,
the cost of finding out is not the constraint.

### Re-frame the claim to what the evidence can carry

Something like: *at 5M tokens, under a frozen linear probe on AG News, no trained condition's
representation exceeded a random-initialised encoder of the same architecture; transfer
decreased monotonically with training budget across all measured conditions and both seeds;
geometric collapse was not the failure mode for the student; and at this scale the recipe is not
separable from the budget and schedule.*

What must **not** be claimed: that random initialisation beats pretraining in general (one task,
one probe, one unpinned baseline); that the latent idea is wrong (it has not been tested at a
budget where it could show); or that D–G identify which component matters (loss only, one seed,
incomparable scales).

### Make one task unable to carry the conclusion

SST-2 is already planned (`README.md:267`, `docs/PROJECT_CONTEXT.md`). Add an **STS-B Spearman
correlation on frozen embeddings** — a regression task where lexical overlap is a weak signal and
word order matters. If the latent conditions beat MLM on STS-B while losing on AG News, that is
a real, publishable interaction and it goes a long way toward explaining the AG News result. If
they lose on both, the negative result is much better supported than it is today.

### State a falsifiable version of the premise

The project's premise is currently unfalsifiable as written, because "masked text is ambiguous"
does not entail "predict the latent". A version that could fail:

> At equal compute and equal head capacity, masked latent prediction yields a higher-rank, more
> linearly readable representation than MLM, and its advantage is concentrated on
> high-ambiguity and rare tokens.

That is testable with the harness that already exists, it can come out false, and — crucially —
the ambiguity analysis already computes the rare/frequent split (`1.523` vs `1.107` nats) needed
to test the second half. A final report built on that sentence has a hypothesis that survived a
real test. The current report has an exploratory grid that produced one robust negative.

---

## 6. Next two weeks, in order

| # | Work | Cost | Why in this position |
|---|---|---|---|
| 1 | P0 corrections (items 1–8): fix the false stopping claim, the false compatibility note, stale dates, stale test count, the missing `requirements.txt` warning, add `scripts/run_sweep.py` + kernel source, wire or delete `span_length` | ~1 day, no GPU | every one of these is a factual error in the repository, and the provenance gap blocks everything downstream |
| 2 | P1 probe redesign (items 9–13): 10-seed baseline band, TF-IDF floor, stratified splits, layer × pooling matrix, params/seconds table. One Kaggle job → `results/person1/probe_v2.json` | ~1 GPU-h | changes what the headline number *means* before more compute is spent on it |
| 3 | P2 training path (items 14–17): shuffling, resampled masks, 80/10/10 option, EMA schedule. Then rerun C/A/B at 5M × 2 seeds | ~2.5 GPU-h | removes the mechanical confounds before scaling |
| 4 | P3 controls (items 18–20): fixed/random-target, head-capacity, teacher-target geometry | ~3 GPU-h | decides whether the negative result is about objectives at all |
| 5 | If (3) shows a real ordering, scale the winning pair to 50M tokens | ~5 GPU-h | only worth spending once the pipeline is trustworthy |
| 6 | D–G with probes and two seeds at the final budget; then report and deck | ~5 GPU-h | D–G cannot support a component claim until they have probes and seeds |

Total: roughly 15 GPU-hours, against 30 per week. The binding constraint is not compute — it is
that steps 1 and 2 must happen before steps 3–6, because every experiment after them inherits
their defects.

---

## 7. Appendix

### 7.1 Every number quoted, and where it came from

Re-derived by script from the cited file; none typed from memory.

| Figure | Value | Source |
|---|---|---|
| Encoder parameters | 11,005,952 | measured: `SmallEncoder(ModelConfig())` |
| Token embedding | 7,813,632 | measured: `m.token_embeddings` |
| Per encoder layer | 789,760 | measured: `m.layers[0]` |
| `MLMHead` | 7,910,458 | measured: `MLMHead(256, 30522)` |
| `AttentionPredictor` | 1,678,080 | measured: `AttentionPredictor(ModelConfig(), 0.1)` |
| `LinearPredictor` | 65,792 | measured: `LinearPredictor(256)` |
| 0.69M tier tokens / updates | 692,919 / 1,217 | `sweep_annealed.json`, all six 0.69M runs |
| 0.69M tier stop reason | `interrupted` | `sweep_annealed.json`, all six 0.69M runs |
| 0.69M shortfall | 7,081 tokens ≈ 12.4 batches | 700,000 − 692,919; ÷ 569.4 tokens/update |
| 5M tier tokens / updates | 5,000,623 / 8,668 | `sweep_annealed.json`, all six 5M runs |
| 5M tier stop reason | `budget` | `sweep_annealed.json`, all six 5M runs |
| Tokens per update | 569.4 (0.69M), 576.9 (5M) | tokens ÷ updates, `sweep_annealed.json` |
| `max_updates` fallback | 10,000 | `src/config.py:83`; no `optimizer` block in any `configs/*.json` |
| LR fraction at 1,217 / 10,000 | 0.8872 | `_schedule_scale` arithmetic, `src/train.py:155-159` |
| EMA mean window | 249 updates (half-life 173) | 0.996/(1−0.996); ln(0.5)/ln(0.996) |
| EMA window as % of run | 20.5% / 2.9% | 249 ÷ 1,217; 249 ÷ 8,668 |
| Probe means, C | 0.5765 → 0.5315 | `sweep_annealed.json`, mean of seeds 17/43 |
| Probe means, A | 0.5895 → 0.5490 | same |
| Probe means, B | 0.5860 → 0.5370 | same |
| Per-seed drops, 0.69M→5M | 0.027 to 0.071 | `sweep_annealed.json`, 6 pairs |
| Anti-collapse drops | 0.013 to 0.047 | `sweep_anticollapse.json`, 6 pairs |
| Degrading cells | 12 of 12 | both sweep files, all (condition, seed) pairs |
| Highest probed student | 0.607 | `sweep_anticollapse.json`, I / seed 43 / 0.69M |
| Trained students probed | 33 | 3 + 6 + 12 + 12 across the four files |
| Random reference | 0.632 / macro-F1 0.6280428897446606 | `probe_agnews_t4.json`, `probe_agnews_long.json` |
| Reference range across project | 0.631–0.632 | 0.631 is `sweep_pass1_no_annealing.json:7` only |
| Majority baseline | 0.274 | all probe and sweep files |
| Probe train class counts | 998 / 797 / 988 / 1217 | `probe_agnews_long.json:train_meta` |
| Probe test class counts | 268 / 274 / 205 / 253 | `probe_agnews_long.json:test_meta` |
| Ambiguity mean / median | 1.277 / 0.602 nats | `ambiguity_wikitext2.json` |
| Positions > log 5 | 30.2% (0.30216) | `ambiguity_wikitext2.json` |
| Span vs independent entropy | 1.633 vs 1.612 nats | `ambiguity_wikitext2.json:policy_comparison` |
| Span-more-ambiguous records | 191 of 400 | `ambiguity_wikitext2.json:policy_comparison` |
| Rare vs frequent entropy | 1.523 vs 1.107 nats | `ambiguity_wikitext2.json:entropy_by_frequency` |
| Geometry rank, C (s17/s43) | 10.23 / 16.95 | `geometry_long.json` |
| Geometry rank, A (s17/s43) | 76.43 / 80.82 | `geometry_long.json` |
| Geometry rank, B (s17/s43) | 67.58 / 72.63 | `geometry_long.json` |
| 5M seconds, A/B/C | 316.0 / 246.2 / 352.8 | `sweep_annealed.json`, mean of 2 seeds |
| 5M tokens/s, A/B/C | 15,825 / 20,311 / 14,176 | tokens ÷ seconds |
| Whole sweep training time | 2,089.4 s = 0.58 h | sum of `seconds`, all 12 runs |
| 50M extrapolation, 12 runs | ~5.1 h | 10 × 5M tier sum (0.51 h) |
| Test suite | 64 passed, 2 skipped (66 collected) | executed: `python3 -m pytest -q` |
| Test functions | 64 | `grep -c "def test_" tests/*.py` |
| Ablation F vs A final loss | 0.9015 vs 0.5052 | `kaggle_long_runs.json:ablations`, `:long` |
| Long-pilot budget | 700,482 tokens / 1,230 updates | `kaggle_long_runs.json:budget` |
| Pilot budget | 22,833 tokens / 3,333 targets | `kaggle_t4_pilot.json:data.counts_per_artifact` |

### 7.2 File:line index of the defects

| # | Defect | Location |
|---|---|---|
| 3.1.1 | False "all 12 stop on budget" claim | `docs/DATA_INTEGRITY.md:20-21` |
| 3.1.1 | Same claim repeated | `docs/PROGRESS.md:132` |
| 3.1.2 | False compatibility note | `src/evaluate_bridge.py:32-36` |
| 3.1.2 | Note injected into every geometry report | `src/evaluate_bridge.py:329` |
| 3.1.2 | Loader that contradicts it | `src/evaluate.py:277-286` |
| 3.1.3 | Stale dates | `docs/PROJECT_CONTEXT.md:7` |
| 3.1.4 | Stale test count | `AGENTS.md:10` |
| 3.1.4 | Stale test count (log banner) | `docs/PROGRESS.md:6` |
| 3.1.5 | Missing claimed warning | `requirements.txt:1-8`; claim at `docs/PROGRESS.md:110` |
| 3.2 | `budget/575` derivation only in prose | `sweep_annealed.json:10`; no code |
| 3.2 | `max_updates` default | `src/config.py:83` |
| 3.2 | Schedule that depends on it | `src/train.py:155-159` |
| 3.3.1 | Corpus-order chunking | `scripts/build_batches.py:196`, `:217-218` |
| 3.3.1 | In-order iteration | `src/train.py:83-99` |
| 3.3.1 | Unused shuffle path | `src/data.py:371`, `:519` |
| 3.3.2 | Masks baked at build time | `scripts/build_batches.py:200-208`, `:325` |
| 3.3.3 | Dead `span_length` | `src/config.py:136`, `:143`; not passed at `scripts/build_batches.py:200-208` |
| 3.3.3 | Defaults that silently apply | `src/data.py:22-23` |
| 3.3.4 | `[MASK]`-only corruption | `src/data.py:108`, `:202` |
| 3.3.5 | Fixed EMA decay | `src/config.py:114`; `configs/*.json` |
| 3.4.1 | Single-draw baseline | `scripts/run_probe.py:147-186`, seed at `:161` |
| 3.4.1 | Misattributed 0.631 | `docs/DATA_INTEGRITY.md:66` |
| 3.4.2 | Final-layer-only probe | `scripts/run_probe.py:92` |
| 3.4.2 | Fixed probe hyperparameters | `src/evaluate.py:334-342` |
| 3.4.3 | No lexical baseline | absent from repo |
| 3.4.4 | Head-of-split probe | `scripts/run_probe.py:46` |
| 3.4.5 | Clean-text-only geometry | `src/evaluate_bridge.py:172-201` |
| 3.4.6 | Ambiguity without the inferential step | `ambiguity_wikitext2.json:limitations` |
| 3.5.1 | Head-capacity asymmetry | `src/objectives.py:21-28` vs `:71-117` |
| 3.5.1 | Second encoder forward | `src/train.py:203-206` |
| 3.5.1 | Asymmetry named but not measured | `docs/PERSON1_LONG_PILOT.md:44-46` |
| 3.6.1 | Unpinned torch | `requirements.txt:1` vs `pyproject.toml` |

### 7.3 What was not inspected or verified

Stated explicitly so no reader assumes more coverage than this audit has:

- **Kaggle run trees.** Not available. These claims in `docs/DATA_INTEGRITY.md` depend on them
  and were **not** independently verified: that all 12 runs carry `git_commit 5f20e6a29c` with
  `dirty=false`; that there are 12 distinct `student.pt` SHA-256 values; that every
  `probe-sweep` entry's `student_sha256` matches the on-disk artefact; that
  `pilot.measured.updates` equals the `metrics.jsonl` line count in all 12. I confirmed only
  that `sweep_annealed.json` and `sweep_anticollapse.json` carry **no per-run commit or hash
  fields**, so those claims cannot be checked from the JSON in this tree at all.
  (`probe_agnews_long.json` and `probe_agnews_t4.json` *do* carry per-student SHA-256, and those
  are present and distinct.)
- **Kaggle kernel source** for `dakshasubramanya/7th-sem-budget-sweep` (v3) and
  `7th-sem-probe-only`. Not in the repository and not retrieved. Everything in §3.2 about *how*
  the sweep was produced is inference from internal consistency of the JSON plus the absence of
  any producer in the tree — not from reading the kernel.
- **The `.pptx` / `.docx` / `.pdf` artefacts** (`docs/Interim_Project_Presentation_7th-sem.pptx`,
  `docs/mid-project-review.pptx`, `docs/SYNOPSIS_interim_draft_paper.docx`,
  `docs/PROJECT_EXPLAINED.pdf`). Their numbers were not cross-checked against the JSON. The
  generators (`scripts/build_midreview_deck.py`, `scripts/build_synopsis.py`,
  `scripts/audit_documents.py`) exist and `audit_documents.py` claims to verify printed numbers,
  but it was not executed here.
- **`results/Multi_doc_compare/`** — an empty directory; nothing to inspect.
- **`docs/assets/logs/`** — read only for `05_geometry.json`, to confirm where the false
  compatibility note propagates. The other logs were not audited.
- **The RNG-parity and resume claims** were not independently exercised beyond running the suite;
  the tests exist and pass (`tests/test_training.py:148`,
  `test_dropout_resume_restores_all_state_optimizer_scheduler_and_rng`).
- **Cross-branch state.** `origin/person1`, `origin/person2`, `origin/person2-fixes` and
  `origin/integration/person1-person2` exist. Only `main` was audited.

### 7.4 Corrections to a claim made in the audit brief

For the record, since this document supersedes the brief it was written from: the brief stated
that `docs/DATA_INTEGRITY.md` §5 documents "0.632 vs 0.631 across probe files". Both probe files
record **0.632**; the 0.631 is in `sweep_pass1_no_annealing.json`, a superseded *sweep* file. The
underlying criticism — that the reference is one unpinned draw — holds and is if anything
strengthened by the correction. The brief also gave the MLM head as ≈7.88M; the measured value is
7,910,458, and the brief's "≈12 GPU-hours" for a 50M grid is ≈5.1 h by the recorded timings, and
its "≈1.2 GPU-hours" for the current 12-run sweep is 0.58 h of logged training seconds.
