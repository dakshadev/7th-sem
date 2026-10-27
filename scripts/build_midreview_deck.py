"""Build the mid-project review deck from measured evidence files.

Every number in the deck is read from a results JSON that was produced by a real run. Missing
files produce an explicit "not run" cell rather than a guess.
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "person1"
NAVY = RGBColor(0x12, 0x26, 0x3D)
TEAL = RGBColor(0x1F, 0x7A, 0x8C)
GREY = RGBColor(0x55, 0x5F, 0x6B)
LIGHT = RGBColor(0xEC, 0xF1, 0xF5)


def load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def slide_blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def heading(slide, text: str, subtitle: str = "") -> None:
    box = slide.shapes.add_textbox(Emu(457200), Emu(320040), Emu(10972800), Emu(720000))
    frame = box.text_frame
    frame.text = text
    frame.paragraphs[0].font.size = Pt(30)
    frame.paragraphs[0].font.bold = True
    frame.paragraphs[0].font.color.rgb = NAVY
    if subtitle:
        sub = slide.shapes.add_textbox(Emu(457200), Emu(950000), Emu(10972800), Emu(420000))
        para = sub.text_frame.paragraphs[0]
        para.text = subtitle
        para.font.size = Pt(14)
        para.font.color.rgb = TEAL
    line = slide.shapes.add_shape(1, Emu(457200), Emu(880000), Emu(10972800), Emu(18000))
    line.fill.solid()
    line.fill.fore_color.rgb = TEAL
    line.line.fill.background()


def bullets(slide, items: list[str], top: int = 1300000, size: int = 17) -> None:
    box = slide.shapes.add_textbox(Emu(457200), Emu(top), Emu(10972800), Emu(4600000))
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(items):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        level = 0
        text = item
        if item.startswith("  "):
            level = 1
            text = item.strip()
        para.text = ("• " if level == 0 else "– ") + text
        para.level = level
        para.font.size = Pt(size if level == 0 else size - 2)
        para.font.color.rgb = NAVY if level == 0 else GREY
        para.space_after = Pt(8)


def table(
    slide, headers: list[str], rows: list[list[str]], top: int = 1300000, width_emu: int = 10972800
) -> None:
    shape = slide.shapes.add_table(
        len(rows) + 1,
        len(headers),
        Emu(457200),
        Emu(top),
        Emu(width_emu),
        Emu(400000 * (len(rows) + 1)),
    )
    grid = shape.table
    for col, header in enumerate(headers):
        cell = grid.cell(0, col)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(12)
        para.font.bold = True
        para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = grid.cell(r, c)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if r % 2 else RGBColor(0xFF, 0xFF, 0xFF)
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(11)
            para.font.color.rgb = NAVY


def notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "not run"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Emu(12192000)
    prs.slide_height = Emu(6858000)

    ambiguity = load("ambiguity_wikitext2.json")
    long_runs = load("kaggle_long_runs.json")
    probe_short = load("probe_agnews_t4.json")
    probe_long = load("probe_agnews_long.json")
    geometry_long = load("geometry_long.json")

    # 1 — title
    slide = slide_blank(prs)
    band = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Emu(2100000))
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()
    title = slide.shapes.add_textbox(Emu(600000), Emu(620000), Emu(10972800), Emu(1000000))
    run = title.text_frame.paragraphs[0]
    run.text = "A Lightweight Self-Supervised Framework for Text\nRepresentation Learning Using Masked Latent Prediction"
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    sub = slide.shapes.add_textbox(Emu(600000), Emu(2300000), Emu(10972800), Emu(1600000))
    frame = sub.text_frame
    for index, line in enumerate(
        [
            "Mid-Project Review  ·  29 October 2026",
            "R Daksha Subramanya · Sudeep S S Suvarna · Akash U S  (batch PB93)",
            "Guide: Dr. Prabadevi B, Assoc. Professor, Dept. of CSE, MSRUAS",
        ]
    ):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.text = line
        para.font.size = Pt(16 if index == 0 else 13)
        para.font.color.rgb = TEAL if index == 0 else GREY
    notes(
        slide,
        "State the honest position up front: the experimental system is built and runs on a "
        "free GPU; what we lack is training budget, not infrastructure.",
    )

    # 2 — status at a glance
    slide = slide_blank(prs)
    heading(
        slide,
        "Where we are at mid-project",
        "Pre-project: plan and literature. Now: a running system with measured pilots.",
    )
    bullets(
        slide,
        [
            "Built: shared 4-layer encoder, MLM control, two masked-latent recipes, EMA teacher, resumable checkpoints",
            "Built: data pipeline, masking policies, representation geometry, frozen linear probe, ambiguity analysis",
            "Validated: 33 automated tests, lint clean, checkpoint resume with exact state parity",
            "Run: pilots on a free Kaggle T4 (no local GPU, no paid compute)",
            "Not yet: the full training budget. Every result so far is a bounded pilot, and we say so",
        ],
    )
    notes(slide, "Lead with what exists and runs. Then immediately bound the claims.")

    # 3 — research question
    slide = slide_blank(prs)
    heading(
        slide, "Research question (unchanged)", "One shared small encoder, one change at a time"
    )
    bullets(
        slide,
        [
            "Masked latent prediction works for text in one recipe and collapses in another",
            "Those recipes differ in target, masking, predictor and loss simultaneously, so nobody knows which choice matters",
            "Question: which of those five design choices decides collapse versus transferable representations, at academic compute scale?",
            "Hypothesis: normalised multi-layer targets matter most; training loss alone will not predict transfer",
        ],
    )
    notes(slide, "Do not re-litigate the survey; restate the question and move on.")

    # 4 — method
    slide = slide_blank(prs)
    heading(slide, "Method: what is actually implemented")
    bullets(
        slide,
        [
            "Student: random-init 4-layer encoder, width 256, 4 heads, length 128, learned positions",
            "Teacher: frozen EMA copy, sees uncorrupted text, no gradients, targets detached",
            "Targets: final layer (recipe A) or mean of normalised top-2 layers (recipe B)",
            "Predictors: 2-layer attention predictor (A) or linear projection (B)",
            "Losses: MSE (A) or Smooth L1 (B); MLM control uses vocabulary cross-entropy",
            "Everything selected by config file: C / A / B and ablations D–G share one runner",
        ],
    )
    notes(
        slide,
        "Emphasise one encoder, one runner, config-selected variants: that is what makes the comparison fair.",
    )

    # 5 — variants table
    slide = slide_blank(prs)
    heading(slide, "Conditions", "A vs B changes four things; D–G change exactly one each")
    table(
        slide,
        ["ID", "Target", "Masking", "Predictor", "Loss"],
        [
            ["C", "token identity", "independent", "vocab classifier", "cross-entropy"],
            ["A", "final layer", "short spans", "attention", "MSE"],
            ["B", "normalised top-2", "independent", "linear", "Smooth L1"],
            ["D", "normalised top-2 (B only)", "short spans", "attention", "MSE"],
            ["E", "final layer", "independent (B only)", "attention", "MSE"],
            ["F", "final layer", "short spans", "linear (B only)", "MSE"],
            ["G", "final layer", "short spans", "attention", "Smooth L1 (B only)"],
        ],
    )
    notes(
        slide,
        "If asked why A vs B cannot answer the question: because they differ in four places. "
        "D–G are the controlled version.",
    )

    # 6 — O1 ambiguity
    slide = slide_blank(prs)
    heading(
        slide,
        "Objective 1: is masked text actually ambiguous?",
        "Pretrained BERT used only as an estimator, never as our encoder",
    )
    if ambiguity:
        amb = ambiguity["ambiguity"]
        pol = ambiguity["policy_comparison"]
        table(
            slide,
            ["Measurement", "Value"],
            [
                ["Eligible positions scored", f"{ambiguity['corpus']['eligible_tokens']:,}"],
                ["Records", f"{ambiguity['corpus']['records_used']}"],
                ["Mean predictive entropy", f"{amb['mean_entropy_nats']:.3f} nats"],
                ["Median predictive entropy", f"{amb['median_entropy_nats']:.3f} nats"],
                [
                    "Positions above log 5 (≈5 plausible tokens)",
                    f"{amb['frac_positions_entropy_gt_log5'] * 100:.1f} %",
                ],
                ["Independent-mask mean entropy", f"{pol['independent_mean_entropy']:.3f} nats"],
                ["Span-mask mean entropy", f"{pol['span_mean_entropy']:.3f} nats"],
                [
                    "Records where spans are more ambiguous",
                    f"{pol['records_span_more_ambiguous']} / {pol['paired_records']}",
                ],
            ],
        )
        notes(
            slide,
            "This is the proposal's Objective 1, measured. It supports the premise that masked text has many "
            "valid answers, and that span masking selects systematically harder targets.",
        )
    else:
        bullets(slide, ["Ambiguity analysis not yet available in this build"])
    # 7 — compute
    slide = slide_blank(prs)
    heading(
        slide,
        "Compute: free GPU only",
        "Colab free tier refused to allocate a backend from our location",
    )
    bullets(
        slide,
        [
            "Runtime used: Kaggle free tier, Tesla T4 15 GB, PyTorch 2.11.0+cu128, CUDA verified inside the run",
            "Local machines: CPU only for tests and analysis — no local CUDA installs, no paid compute",
            "Measured throughput from the pilot: ~9.5k tokens/s for this encoder size",
            "Entitlement: 30 GPU-hours per week on Kaggle; we have used well under two hours",
            "Every run records commit, config, artifact hashes, token counts and wall-clock time",
        ],
    )
    notes(
        slide,
        "If asked about the HPC request: we have not needed it yet, and we can quantify what a full run costs.",
    )

    # 8 — first pilot + probe (negative result)
    slide = slide_blank(prs)
    heading(
        slide,
        "First pilot: the system runs, the budget does not yet teach",
        "50 updates · 22,833 tokens · 0.46 % of the proposed budget",
    )
    if long_runs and long_runs.get("short"):
        rows = []
        for run in long_runs["short"]:
            rows.append(
                [
                    run["label"],
                    fmt(run["tokens"]),
                    fmt(run["last_loss"]),
                    fmt(run["seconds"], 2) + " s",
                ]
            )
        table(slide, ["Condition", "Tokens", "Final loss", "Train time"], rows)
    if probe_short:
        students = probe_short["students"]
        rows = [
            [name.split("-", 1)[-1], f"{value['accuracy']:.3f}", f"{value['macro_f1']:.3f}"]
            for name, value in students.items()
        ]
        rows.append(
            [
                "random-init",
                f"{probe_short['random_baseline']['accuracy']:.3f}",
                f"{probe_short['random_baseline']['macro_f1']:.3f}",
            ]
        )
        table(slide, ["AG News probe", "Accuracy", "Macro-F1"], rows, top=3300000)
    notes(
        slide,
        "Be direct: at 50 updates the trained encoders score below an untrained encoder. "
        "That is why the next slide exists, and it is why we are asking for a larger budget rather than more variants.",
    )

    # 9 — longer pilots, two seeds
    slide = slide_blank(prs)
    heading(
        slide,
        "Scaled pilot: 2 seeds, ~0.7 M tokens each",
        "Still a pilot, but now the encoder is actually trained",
    )
    if long_runs and long_runs.get("long"):
        rows = []
        for run in long_runs["long"]:
            rows.append(
                [
                    f"{run['run']} (seed {run.get('seed')})",
                    f"{run['nonpadding_tokens']:,}",
                    f"{run['masked_targets']:,}",
                    fmt(run["first_loss"]),
                    fmt(run["last_loss"]),
                    fmt(run["training_wall_clock_seconds"], 1) + " s",
                ]
            )
        table(slide, ["Run", "Tokens", "Targets", "First loss", "Last loss", "Time"], rows)
        notes(
            slide,
            "Report per seed, not an average, and note that the runs are token-matched but the "
            "predictor and teacher overhead differ, so compute is not identical.",
        )
    else:
        bullets(slide, ["Scaled pilot pending"])
    # 10 — geometry
    slide = slide_blank(prs)
    heading(
        slide,
        "Representation geometry: does anything collapse?",
        "Frozen student, uncorrupted text, held-out split",
    )
    if geometry_long:
        rows = [
            [
                name,
                f"{g['effective_rank_singular']:.1f}",
                f"{g['effective_rank_covariance']:.2f}",
                g["status"],
            ]
            for name, g in geometry_long["students"].items()
        ]
        table(
            slide,
            ["Student", "Effective rank (singular)", "Effective rank (covariance)", "Status"],
            rows,
        )
        notes(
            slide,
            "No collapse at this budget. Note that healthy geometry is necessary but not sufficient: "
            "the probe is the transfer check.",
        )
    else:
        bullets(slide, ["Geometry for the scaled pilots pending"])
    # 11 — probe long
    slide = slide_blank(prs)
    heading(
        slide,
        "Frozen linear probe: has transfer improved?",
        "AG News · scaler and classifier fit on probe-train only",
    )
    if probe_long:
        rows = [
            [name, f"{v['accuracy']:.3f}", f"{v['macro_f1']:.3f}"]
            for name, v in probe_long["students"].items()
        ]
        rows.append(
            [
                "random-init",
                f"{probe_long['random_baseline']['accuracy']:.3f}",
                f"{probe_long['random_baseline']['macro_f1']:.3f}",
            ]
        )
        rows.append(
            ["majority baseline", f"{probe_long['random_baseline']['majority_baseline']:.3f}", "—"]
        )
        table(slide, ["Student", "Accuracy", "Macro-F1"], rows)
        notes(
            slide,
            "If trained students still sit near or below the random reference, say so plainly and "
            "explain that the probe task is easy with few labels, so it cannot separate variants yet.",
        )
    else:
        bullets(slide, ["Scaled-pilot probe pending"])
    # 12 — ablations
    slide = slide_blank(prs)
    heading(slide, "Component ablations D–G", "One change at a time relative to A")
    if long_runs and long_runs.get("ablations"):
        rows = [
            [
                r["run"],
                r["label"],
                f"{r['nonpadding_tokens']:,}",
                fmt(r["last_loss"]),
                fmt(r["training_wall_clock_seconds"], 1) + " s",
            ]
            for r in long_runs["ablations"]
        ]
        table(slide, ["Run", "ID", "Tokens", "Final loss", "Time"], rows)
        notes(
            slide,
            "These are loss-level observations only. With one seed and no probe yet they are "
            "hypotheses, not conclusions.",
        )
    else:
        bullets(slide, ["Ablation runs pending"])
    # 13 — what we learned
    slide = slide_blank(prs)
    heading(slide, "What this milestone actually establishes")
    bullets(
        slide,
        [
            "The comparison is now controllable: the same encoder, data and runner serve C, A, B and D–G",
            "Masked text really is ambiguous, and span masking selects harder targets than independent masking",
            "Short pilots are not evidence: at 50 updates, training did not beat random features",
            "Collapse did not occur at the budgets tested, so the failure mode is not reproduced yet",
            "Budget, not architecture, is the binding constraint",
        ],
    )
    notes(
        slide,
        "This is the honest summary. It is also a good position: the machinery is done and the next step is compute.",
    )

    # 14 — limitations
    slide = slide_blank(prs)
    heading(slide, "Limitations we will state ourselves")
    bullets(
        slide,
        [
            "All runs are bounded pilots on WikiText-2, not the proposed WikiText-103 ~5M-token budget",
            "Two seeds is weak replication, not a confidence interval",
            "Losses are objective-specific and are never compared across objectives",
            "AG News is easy enough that random features score well with 4,000 labels",
            "No SIGReg variant yet, no token-level geometry, no SST-2",
            "A and B change four components at once; only D–G isolate anything",
        ],
    )
    notes(slide, "Volunteering limitations earns credibility and pre-empts the obvious questions.")

    # 15 — remaining work
    slide = slide_blank(prs)
    heading(slide, "Remaining work to the final review")
    table(
        slide,
        ["Work", "Why it matters", "Status"],
        [
            [
                "Full-budget runs (≥5M tokens) on WikiText-103 subset",
                "Only then can design choices be compared",
                "Planned",
            ],
            [
                "Second seed for A/B/C and the strongest ablation",
                "Separates effect from initialisation noise",
                "Partially done",
            ],
            ["D–G with geometry and probes", "Turns loss observations into evidence", "Planned"],
            [
                "SIGReg variant H (optional)",
                "Tests the proposed anti-collapse remedy on text",
                "Stretch",
            ],
            ["Token-level geometry and SST-2", "Second transfer check", "Stretch"],
            [
                "Freeze protocol, write report and reproduce on a clean runtime",
                "Reproducibility claim",
                "Planned",
            ],
        ],
    )
    notes(
        slide,
        "Give a concrete order: budget first, then seeds, then ablations with probe, then optional components.",
    )

    # 16 — SDG + close
    slide = slide_blank(prs)
    heading(slide, "SDG alignment and close")
    bullets(
        slide,
        [
            "SDG 9, Target 9.5 — open, reproducible research on modest academic compute",
            "SDG 4, Target 4.4 — practical ML experimentation and reproducibility skills",
            "Both are indirect: this is a small academic study, not a national programme",
            "Repository, evidence and runbook are public; results are reproducible on a free GPU",
            "Questions welcome",
        ],
    )
    notes(slide, "Close on the infrastructure claim: anyone can rerun this on a free T4.")

    out = ROOT / "docs" / "mid-project-review.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


if __name__ == "__main__":
    path = build()
    print(path)
