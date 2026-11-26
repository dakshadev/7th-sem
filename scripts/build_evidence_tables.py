"""Build the parameter and measured-cost tables the results JSON never showed.

Every number is either measured from the real modules (parameter counts), read from the
sweep JSON (seconds, tokens, probe accuracy), or a stated arithmetic combination of the two
(tokens/second, mean over seeds). Nothing is estimated.

Usage:
    python scripts/build_evidence_tables.py                 # print Markdown
    python scripts/build_evidence_tables.py --out docs/EVIDENCE_TABLES.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.model import SmallEncoder
from src.objectives import AttentionPredictor, LinearPredictor, MLMHead

CONFIG_FOR_LABEL = {
    "C": "configs/mlm_c.json",
    "A": "configs/latent_a.json",
    "B": "configs/latent_b.json",
    "D": "configs/ablation_d.json",
    "E": "configs/ablation_e.json",
    "F": "configs/ablation_f.json",
    "G": "configs/ablation_g.json",
    "H": "configs/latent_h_vicreg.json",
    "I": "configs/latent_i_barlow.json",
    "J": "configs/latent_j_noema.json",
    "K": "configs/latent_k_constant.json",
    "L": "configs/latent_l_noise.json",
    "M": "configs/mlm_m_small_head.json",
    "A2": "configs/latent_a2_v2.json",
    "B2": "configs/latent_b2_v2.json",
    "C2": "configs/mlm_c2_v2.json",
}
DEFAULT_RESULTS = (
    "results/person1/sweep_annealed.json",
    "results/person1/sweep_anticollapse.json",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parameter_counts(label: str, root: Path | None = None) -> dict[str, Any]:
    """Encoder, extra head and total trainable parameters for one condition."""
    base = root or repo_root()
    config = load_config(base / CONFIG_FOR_LABEL[label])
    encoder = sum(item.numel() for item in SmallEncoder(config.model).parameters())
    if config.objective.kind == "mlm":
        # Honour the configured bottleneck: M's head is not C's head, and reporting the full
        # 7.9M for M would defeat the purpose of the control.
        head = sum(
            item.numel()
            for item in MLMHead(
                config.model.width,
                config.model.vocab_size,
                config.objective.mlm_head_width,
            ).parameters()
        )
        head_name = "MLMHead" if config.objective.mlm_head_width is None else "MLMHead(reduced)"
    elif config.objective.predictor == "attention":
        head = sum(
            item.numel()
            for item in AttentionPredictor(
                config.model, config.objective.predictor_dropout
            ).parameters()
        )
        head_name = "AttentionPredictor"
    else:
        head = sum(item.numel() for item in LinearPredictor(config.model.width).parameters())
        head_name = "LinearPredictor"
    return {
        "label": label,
        "head_name": head_name,
        "encoder": encoder,
        "head": head,
        "total": encoder + head,
        "teacher_forward": config.objective.kind == "latent" and config.teacher.mode == "ema",
    }


def load_runs(paths: list[str], root: Path | None = None) -> list[dict[str, Any]]:
    """Records that can populate the cost table: needs a budget, seconds and an accuracy."""
    base = root or repo_root()
    runs: list[dict[str, Any]] = []
    for name in paths:
        path = base / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("runs"), list):
            runs.extend(payload["runs"])
    return runs


def load_executed_labels(root: Path | None = None) -> dict[str, int]:
    """Count every recorded run per condition, across both result-file schemas.

    `kaggle_long_runs.json` predates the `runs` schema: it splits its records across `long`
    and `ablations` and carries no probe numbers. Those records cannot enter the cost table,
    but D-G did run there, so a "never run" marker must not be printed for them.

    The superseded pass is skipped: `docs/DATA_INTEGRITY.md` forbids pooling across experiment
    passes, and its four extra A/B/C runs would otherwise double those counts.
    """
    base = root or repo_root()
    counts: dict[str, int] = {}
    for path in sorted((base / "results" / "person1").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("status", "").startswith("SUPERSEDED"):
            continue
        records = list(payload.get("runs") or [])
        for section in ("long", "ablations"):
            records.extend(payload.get(section) or [])
        for item in records:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if isinstance(label, str) and label:
                counts[label] = counts.get(label, 0) + 1
    return counts


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["label"], int(run["budget_tokens"])), []).append(run)
    rows: list[dict[str, Any]] = []
    for (label, budget), items in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        seconds = [float(item["seconds"]) for item in items if item.get("seconds") is not None]
        tokens = [
            int(item["nonpadding_tokens"])
            for item in items
            if item.get("nonpadding_tokens") is not None
        ]
        accuracy = [
            float(item["probe_accuracy"])
            for item in items
            if item.get("probe_accuracy") is not None
        ]
        rows.append(
            {
                "label": label,
                "budget": budget,
                "runs": len(items),
                "seconds": mean(seconds),
                "tokens_per_second": (sum(tokens) / sum(seconds))
                if seconds and tokens and sum(seconds) > 0
                else None,
                "accuracy": mean(accuracy),
                "accuracy_n": len(accuracy),
            }
        )
    return rows


def _number(value: float | None, digits: int = 3) -> str:
    return "not measured" if value is None else f"{value:.{digits}f}"


def render(runs: list[dict[str, Any]], labels: list[str]) -> str:
    # Every configured condition gets a parameter row, whether or not it has a run: parameter
    # counts are measured from the modules and need no execution. A condition with no runs is
    # still useful here — M's whole purpose is its head size.
    configured = sorted(CONFIG_FOR_LABEL)
    measured = sorted({run["label"] for run in runs})
    remaining = [name for name in configured if name not in measured]
    params = [parameter_counts(label) for label in [*measured, *remaining]]
    cost = summarise_runs(runs)
    lines = [
        "# Parameter and measured-cost tables",
        "",
        "Generated by `scripts/build_evidence_tables.py` from `results/person1/*.json`.",
        "Parameter counts are measured from the real modules; seconds, tokens and probe",
        "accuracy are read from the sweep JSON; tokens/second and seed means are stated",
        "arithmetic over those fields. Cells with no measurement are marked, never filled.",
        "The 'Runs recorded' column counts every recorded run of that condition across all",
        "result files except the superseded pass, including the loss-only runs that carry no",
        "probe number and therefore cannot appear in the cost table below.",
        "",
        "## Trainable parameters by condition",
        "",
        "| Condition | Encoder | Extra head | Extra head params | Trainable total "
        "| Teacher forward | Runs recorded |",
        "|---|---|---|---|---|---|---|",
    ]
    run_counts: dict[str, int] = load_executed_labels()
    for item in params:
        count = run_counts.get(item["label"], 0)
        lines.append(
            f"| {item['label']} | {item['encoder']:,} | {item['head_name']} | {item['head']:,} "
            f"| {item['total']:,} | {'yes' if item['teacher_forward'] else 'no'} "
            f"| {count if count else 'none — never run'} |"
        )
    lines.extend(
        [
            "",
            "The encoder is identical in every condition. The extra head is not: the MLM",
            "head is a 256 -> 30522 projection, while B's linear predictor has no hidden",
            "layer. Comparisons of accuracy and loss must be read with that in mind.",
            "",
            "## Measured cost and frozen-probe accuracy",
            "",
            "| Condition | Budget tokens | Runs | Seconds (mean) | Tokens/s "
            "| Probe accuracy (mean) | Probed runs |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in cost:
        throughput = (
            "not measured"
            if row["tokens_per_second"] is None
            else f"{row['tokens_per_second']:,.0f}"
        )
        lines.append(
            f"| {row['label']} | {row['budget']:,} | {row['runs']} "
            f"| {_number(row['seconds'], 1)} | {throughput} "
            f"| {_number(row['accuracy'], 4)} | {row['accuracy_n']}/{row['runs']} |"
        )
    lines.extend(
        [
            "",
            "Accuracy is the frozen AG News linear probe (4,000 train / 1,000 held-out).",
            "It is not comparable to the training losses in the same JSON: those are",
            "objective-specific. A row with fewer probed runs than runs was not fully probed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", default=list(DEFAULT_RESULTS))
    parser.add_argument("--out", help="write Markdown here instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> str:
    args = build_parser().parse_args(argv)
    runs = load_runs(args.results)
    if not runs:
        raise SystemExit("no runs found in the requested result files")
    markdown = render(runs, sorted({run["label"] for run in runs}))
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
        print(f"wrote {destination}")
    else:
        print(markdown)
    return markdown


if __name__ == "__main__":
    main()
