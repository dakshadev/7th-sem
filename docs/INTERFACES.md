# Shared interfaces — proposed v1

Both owners should confirm this contract before coding. Breaking changes require a coordinated update to both implementations and this document. Shapes below describe the intended interface, not existing classes.

## Batch contract

A dictionary of PyTorch tensors on CPU from the loader; the runner moves them to its device:

| Field | Shape / dtype | Meaning |
|---|---|---|
| input_ids | [B,L], int64 | Original uncorrupted tokenizer IDs |
| masked_input_ids | [B,L], int64 | Student's corrupted input |
| attention_mask | [B,L], bool | True = valid token; false = padding |
| prediction_mask | [B,L], bool | True = eligible selected position contributing to loss |

L=128 initially. prediction_mask must be false at padding and all tokenizer special-token positions. IDs must be within tokenizer vocabulary. MLM labels are derived from input_ids at prediction_mask; no second conflicting label source. A batch with no eligible targets must be safely skipped/reported or rejected, never produce NaN from empty reduction.

Proposed simplest initial corruption: replace selected eligible positions with tokenizer.mask_token_id. Do not silently assume BERT 80/10/10 corruption. If another policy is selected, record it and use it consistently in matched comparisons. Mask ratio is approximately 15% of eligible positions. Person 2 must specify independent sampling versus fixed counts, span length/distribution and boundary handling. Evaluation masks must be deterministic; training uses controlled randomness.

## Data isolation

Split before packing/tokenization boundaries can leak content. Prefer canonical source train/validation/test splits; train only on training material. Record dataset ID/config/revision, subset-selection procedure, tokenizer revision, document-boundary/packing policy, eligible-token counts and hashes. Report tokens processed with an explicit counting convention (nonpadding original student input tokens; separately log masked targets). For the pilot use deterministic truncation/padding within records/documents rather than silently joining splits or unrelated records.

## Encoder contract

Proposed `forward(input_ids, attention_mask, return_hidden_states=False)` returns:
- `last_hidden_state`: [B,L,256].
- `hidden_states`: four post-encoder-layer tensors [B,L,256] when requested; do not include embedding-only output in the top-2 average.

Document attention-mask polarity conversion at the PyTorch attention boundary. Student receives masked_input_ids; teacher receives input_ids. Teacher excludes MLM head/predictor, starts as a copy of the student encoder, has gradients disabled, uses evaluation mode for targets and updates by EMA only after a successful optimizer update. Stop gradients through targets. Record the EMA coefficient/schedule. Prediction heads/normalizers and their train/eval behavior must be explicit.

## Evaluation contract

Frozen student encoder in evaluation mode receives **uncorrupted** text. Use attention-mask-aware mean pooling excluding padding and special tokens for sentence vectors [N,256]. Person 2 should use tokenizer special-token metadata to construct the pooling mask; do not assume attention_mask alone excludes CLS/SEP. Guard empty examples.

Use fixed evaluation examples and record counts. If token geometry is also computed, label it separately from sentence geometry. Return named numeric metrics plus status for undefined/degenerate cases. Evaluation must not mutate encoder weights. Probe labels are used only for probe fitting/evaluation, not pretraining. Fit scaling and classifier on probe training data only.

## Artifact boundary

A checkpoint must expose student encoder weights and model configuration so Person 2 can load it without constructing a teacher. Resumable training checkpoint also includes prediction heads, teacher (latent runs), optimizer, scheduler/AMP state if used, step/token counters, random-generator states, sampler/data position or a documented resume limitation. Define and test a checkpoint schema/version jointly.

Run metadata: run ID, variant, seed, Git commit, dirty flag/diff reference, complete config, dataset/tokenizer manifests, versions, hardware, elapsed training time, tokens, output locations and completion/failure status. Machine-local paths are not collaborator-accessible artifact handles.
