"""Objective 1 — corpus profile and masked-word ambiguity analysis.

Implements the proposal's Objective 1 with an explicitly separate, labelled use of a *pretrained*
BERT model: BERT is used only as an ambiguity estimator over the corpus (predictive entropy at
each position). It contributes no weights to the project's encoder.

Outputs a JSON record plus plots. Nothing here is a training result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

PILOT_CONFIG = "wikitext-2-raw-v1"
DATASET_NAME = "Salesforce/wikitext"
BERT_MODEL = "google-bert/bert-base-uncased"


def load_texts(
    limit: int, min_chars: int, dataset: str, config_name: str, split: str
) -> tuple[list[str], dict]:
    from datasets import load_dataset

    data = load_dataset(dataset, config_name, split=split)
    texts: list[str] = []
    for row in data:
        text = str(row.get("text", "")).strip()
        if len(text) >= min_chars:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts, {
        "dataset": dataset,
        "config": config_name,
        "split": split,
        "requested": limit,
        "used": len(texts),
    }


@torch.no_grad()
def position_entropies(
    texts: list[str],
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
    device: torch.device,
    max_positions_per_record: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Per-position predictive entropy with the surrounding context intact.

    For every eligible position we mask ONLY that position and score the model on the rest of
    the sequence. Masking all positions at once would remove the context and return a
    near-uniform distribution, which measures nothing.
    """
    entropies: list[torch.Tensor] = []
    top1s: list[torch.Tensor] = []
    alternatives: list[torch.Tensor] = []
    reserved = {int(i) for i in tokenizer.all_special_ids}
    mask_id = tokenizer.mask_token_id

    for text in texts:
        encoded = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
        ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device).bool()
        positions = [
            int(i)
            for i in range(ids.shape[1])
            if bool(attention[0, i]) and int(ids[0, i]) not in reserved
        ]
        if max_positions_per_record and len(positions) > max_positions_per_record:
            step = len(positions) / max_positions_per_record
            positions = [positions[int(i * step)] for i in range(max_positions_per_record)]
        if not positions:
            continue
        ent = torch.empty(len(positions))
        top1 = torch.empty(len(positions))
        alt = torch.empty(len(positions))
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            batch_ids = ids.expand(len(chunk), -1).clone()
            batch_att = attention.expand(len(chunk), -1)
            for row, pos in enumerate(chunk):
                batch_ids[row, pos] = mask_id
            logits = model(input_ids=batch_ids, attention_mask=batch_att).logits
            for row, pos in enumerate(chunk):
                probs = functional.softmax(logits[row, pos].float(), dim=-1)
                ent[start + row] = -(probs * probs.clamp_min(1e-12).log()).sum()
                top1[start + row] = probs.max()
                alt[start + row] = (probs > 0.1).sum()
        entropies.append(ent.cpu())
        top1s.append(top1.cpu())
        alternatives.append(alt.cpu())
    return entropies, top1s, alternatives


def select_positions(
    entropy_len: int, ratio: float, policy: str, rng: np.random.Generator, span_len: int = 3
) -> list[int]:
    """Mirror the two masking policies on a sequence of `entropy_len` eligible positions."""
    target = max(1, round(entropy_len * ratio))
    if policy == "independent":
        chosen = rng.choice(entropy_len, size=min(target, entropy_len), replace=False)
        return sorted(int(i) for i in chosen)
    chosen: list[int] = []
    taken = set()
    attempts = 0
    while len(chosen) < target and attempts < 200:
        attempts += 1
        start = int(rng.integers(0, entropy_len))
        for offset in range(span_len):
            idx = start + offset
            if idx >= entropy_len or idx in taken:
                break
            taken.add(idx)
            chosen.append(idx)
            if len(chosen) >= target:
                break
    return sorted(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Objective 1: corpus profile and masked-word ambiguity."
    )
    parser.add_argument("--max-records", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-positions-per-record",
        type=int,
        default=0,
        help="0 = score every eligible position; otherwise sample this many per record",
    )
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--span-length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--plot-prefix", default=None)
    args = parser.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL).to(device).eval()

    texts, meta = load_texts(args.max_records, 30, DATASET_NAME, PILOT_CONFIG, "train")
    entropies, top1s, alternatives = position_entropies(
        texts,
        tokenizer,
        model,
        args.max_length,
        args.batch_size,
        device,
        args.max_positions_per_record,
    )

    per_record = []
    for ent, t1, alt in zip(entropies, top1s, alternatives, strict=True):
        n = int(ent.numel())
        if n == 0:
            continue
        indep = select_positions(n, args.mask_ratio, "independent", rng)
        span = select_positions(n, args.mask_ratio, "span", rng, args.span_length)
        per_record.append(
            {
                "eligible_tokens": n,
                "mean_entropy": float(ent.mean()),
                "median_entropy": float(ent.median()),
                "frac_high_ambiguity": float((ent > math.log(5)).float().mean()),
                "independent_mean_entropy": float(ent[indep].mean()) if indep else None,
                "span_mean_entropy": float(ent[span].mean()) if span else None,
                "independent_mean_top1": float(t1[indep].mean()) if indep else None,
                "span_mean_top1": float(t1[span].mean()) if span else None,
                "independent_mean_alternatives": float(alt[indep].mean()) if indep else None,
                "span_mean_alternatives": float(alt[span].mean()) if span else None,
            }
        )

    all_entropy = torch.cat(entropies).numpy()
    indep_e = np.array(
        [
            r["independent_mean_entropy"]
            for r in per_record
            if r["independent_mean_entropy"] is not None
        ]
    )
    span_e = np.array(
        [r["span_mean_entropy"] for r in per_record if r["span_mean_entropy"] is not None]
    )
    paired = indep_e.shape == span_e.shape
    diff = span_e - indep_e if paired else None

    # token-frequency relationship
    token_counts = Counter()
    token_entropy: dict[int, list[float]] = defaultdict(list)
    for text, ent in zip(texts, entropies, strict=True):
        ids = tokenizer(text, max_length=args.max_length, truncation=True)["input_ids"]
        reserved = set(int(i) for i in tokenizer.all_special_ids)
        kept = [t for t in ids if t not in reserved]
        values = ent.tolist()
        for token_id, value in zip(kept, values, strict=False):
            token_counts[token_id] += 1
            token_entropy[token_id].append(value)
    buckets = {"rare(1-2)": [], "mid(3-20)": [], "frequent(>20)": []}
    for token_id, values in token_entropy.items():
        count = token_counts[token_id]
        key = "rare(1-2)" if count <= 2 else ("mid(3-20)" if count <= 20 else "frequent(>20)")
        buckets[key].extend(values)

    report = {
        "schema_version": 1,
        "objective": "O1 — corpus profile and masked-word ambiguity",
        "note": (
            "Pretrained BERT (google-bert/bert-base-uncased) is used ONLY as an ambiguity "
            "estimator; its weights are never used by the project encoder. Entropy is in nats "
            "per position with all other real tokens visible (single-position masking)."
        ),
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "estimator_model": BERT_MODEL,
            "estimator_revision": getattr(model.config, "_name_or_path", BERT_MODEL),
        },
        "data": meta,
        "settings": {
            "max_length": args.max_length,
            "mask_ratio": args.mask_ratio,
            "span_length": args.span_length,
            "seed": args.seed,
            "max_positions_per_record": args.max_positions_per_record or "all",
        },
        "corpus": {
            "records_used": len(per_record),
            "eligible_tokens": int(sum(r["eligible_tokens"] for r in per_record)),
            "unique_tokens": len(token_counts),
            "mean_tokens_per_record": float(np.mean([r["eligible_tokens"] for r in per_record])),
        },
        "ambiguity": {
            "mean_entropy_nats": float(all_entropy.mean()),
            "median_entropy_nats": float(np.median(all_entropy)),
            "p90_entropy_nats": float(np.percentile(all_entropy, 90)),
            "frac_positions_entropy_gt_log5": float((all_entropy > math.log(5)).mean()),
            "frac_positions_entropy_lt_1nat": float((all_entropy < 1.0).mean()),
            "entropy_by_frequency": {
                key: {"positions": len(values), "mean_entropy": float(np.mean(values))}
                for key, values in buckets.items()
                if values
            },
        },
        "policy_comparison": {
            "independent_mean_entropy": float(indep_e.mean()) if indep_e.size else None,
            "span_mean_entropy": float(span_e.mean()) if span_e.size else None,
            "paired_records": int(indep_e.size),
            "mean_difference_span_minus_independent": float(diff.mean())
            if diff is not None and diff.size
            else None,
            "records_span_more_ambiguous": int((diff > 0).sum()) if diff is not None else None,
            "records_independent_more_ambiguous": int((diff < 0).sum())
            if diff is not None
            else None,
        },
        "limitations": [
            "Ambiguity comes from one pretrained model's distribution, not the project encoder.",
            "WikiText-2 pilot subset, not the WikiText-103 target corpus.",
            "Single-position masking approximates, but does not reproduce, joint span masking.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.plot_prefix:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        prefix = Path(args.plot_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(all_entropy, bins=60, color="#3b6ea5")
        axes[0].set_title("Predictive entropy per position")
        axes[0].set_xlabel("nats")
        keys = list(buckets)
        axes[1].bar(keys, [np.mean(buckets[k]) for k in keys], color="#8a5a3b")
        axes[1].set_title("Mean entropy by token frequency")
        axes[1].set_ylabel("nats")
        axes[2].hist(indep_e, bins=30, alpha=0.6, label="independent", color="#3b6ea5")
        axes[2].hist(span_e, bins=30, alpha=0.6, label="span", color="#b5563b")
        axes[2].set_title("Selected-position entropy by policy")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(f"{prefix}.png", dpi=130)
        plt.close(fig)

    print(
        json.dumps(
            {k: report[k] for k in ("data", "corpus", "ambiguity", "policy_comparison")}, indent=2
        )
    )
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
