"""Second pass on the deck: the mentor's follow-up asks.

- module-wise result screenshots, as bullets, as implementation evidence
- a hyperparameter and training-parameter table
- a hyperlink to the GitHub repository
- a dashboard screenshot and a declared slot for the demo video
- slide 11 (methodology) in table format
- a bullet character on every content paragraph, since she asked for bullets on all slides
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Inches, Pt

ROOT = Path("C:/Users/daksh/MyLife/Work/7th-sem")
DECK = ROOT / "docs" / "Interim_Project_Presentation_7th-sem.pptx"
ASSETS = ROOT / "docs" / "assets"
REPO = "https://github.com/dakshadev/7th-sem"

A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

OUTLINE = [
    "Introduction — why latent targets, and why small budgets change the answer",
    "Problem statement — the call, and the research question",
    "Literature survey — 15 papers in three tables, and the approved gaps G1–G5",
    "Approved objectives, and progress against each one",
    "Methodology — five modules, and the hyperparameters they run on",
    "Dataset description and computing requirements",
    "Implementation evidence — module-wise results, then the preliminary results",
    "Live demonstration — dashboard and repository walkthrough",
    "Remaining work, risks and SDG alignment",
    "Conclusion and references",
]

METHOD_ROWS = [
    ["Module", "What it does", "Implementation"],
    ["1 · Data pipeline", "Loads WikiText, tokenises, masks 15% under a span or independent policy, and "
     "writes a manifested batch file with a hash", "src/data.py, scripts/build_batches.py"],
    ["2 · Encoder", "4-layer Transformer, width 256, shared by every condition so only the objective "
     "changes", "src/model.py"],
    ["3 · Objectives", "MLM control, latent recipes A and B, VICReg and Barlow auxiliaries, and a "
     "no-teacher control", "src/objectives.py, configs/*.json"],
    ["4 · Training", "AdamW with an annealed schedule, EMA teacher, resumable checkpoints and "
     "provenance in every run", "src/train.py, scripts/run_pilot.py"],
    ["5 · Evaluation", "Effective rank, frozen linear probe on AG News, and the ambiguity analysis",
     "src/evaluate.py, scripts/run_probe.py"],
]

HYPER_ROWS = [
    ["Parameter", "Value", "Parameter", "Value"],
    ["Encoder layers", "4", "Encoder width", "256"],
    ["Attention heads", "4", "Feed-forward width", "1024"],
    ["Dropout", "0.10", "Sequence length", "128"],
    ["Vocabulary", "30,522 (bert-base-uncased)", "Positional encoding", "learned"],
    ["Optimiser", "AdamW", "Learning rate", "3e-4"],
    ["Weight decay", "0.01", "Betas", "(0.9, 0.999)"],
    ["Warmup", "100 updates", "Gradient clipping", "1.0"],
    ["Schedule", "linear decay to 0", "Max updates", "budget / 575"],
    ["Batch size", "8 sequences", "Tokens per update", "575 (measured)"],
    ["Mask ratio", "15% of positions", "Mask policy", "span-3 or independent"],
    ["Teacher", "EMA copy, decay 0.996", "Teacher update", "after each step, detached"],
    ["Budget tier 1", "0.69M tokens (1,217 updates)", "Budget tier 2", "5.0M tokens (8,668 updates)"],
    ["Seeds", "17 and 43", "Runs reported", "24 (6 conditions × 2 budgets × 2 seeds)"],
]

EVIDENCE_BULLETS = [
    "Data module: 600 WikiText-103 records became 75 batches — 39,388 tokens, 5,767 masked targets, "
    "hashed and manifested before training.",
    "Training module: latent recipe A on CPU, loss 2.018 → 1.086 over 75 updates, with checkpoint, "
    "metrics and metadata written.",
    "Same data, same schedule, MLM control — the only difference is what the model is asked to predict.",
    "64 automated tests pass, and every run records its commit hash and the tokens it actually saw.",
]

DEMO_BULLETS = [
    "Dashboard rebuilt from results/person1/*.json: six conditions, two budgets, geometry, ambiguity.",
    "Walkthrough: build batches → train 40 updates → probe → geometry, all from committed scripts.",
    "Repository: github.com/dakshadev/7th-sem (link on this slide and on the contents slide).",
    "Demo recording: see the repository's docs/assets folder — the file is added after recording, not "
    "reported as done here.",
]


def add_bullets(prs) -> int:
    """Every content paragraph gets a real bullet character; sub-headings and tables are skipped."""
    changed = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            if shape.is_placeholder and shape.placeholder_format.idx == 0:
                continue
            for paragraph in shape.text_frame.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                properties = paragraph._p.get_or_add_pPr()
                if properties.find(f"{A_NS}buChar") is not None:
                    continue
                if properties.find(f"{A_NS}buNone") is not None:
                    continue
                if text[0].isdigit() and text[1:3] == ". ":
                    continue
                properties.set("marL", "285750")
                properties.set("indent", "-285750")
                bullet = OxmlElement("a:buChar")
                bullet.set("char", "•")
                properties.append(bullet)
                changed += 1
    return changed


def clone(prs, model_index: int):
    """A fresh slide on the same layout.

    Shapes are deliberately not copied: deep-copying them pulls relationship ids into a part that
    has no such relationship, and PowerPoint then refuses the file. The layout already carries the
    title and body placeholders, so the slide arrives correctly styled and positioned.
    """
    model = prs.slides[model_index]
    layout = model.slide_layout
    slide = prs.slides.add_slide(layout)
    return slide


def placeholder(slide, idx: int):
    return next(sh for sh in slide.shapes if sh.is_placeholder and sh.placeholder_format.idx == idx)


def set_title(slide, text: str) -> None:
    frame = placeholder(slide, 0).text_frame
    first = frame.paragraphs[0]
    runs = list(first.runs)
    if runs:
        runs[0].text = text
        for extra in runs[1:]:
            extra._r.getparent().remove(extra._r)
    else:
        first.add_run().text = text


def set_body(slide, lines, size: float = 12.5) -> None:
    frame = placeholder(slide, 1).text_frame
    frame.word_wrap = True
    modelled = deepcopy(frame._txBody.findall(f"{A_NS}p")[0])
    body = frame._txBody
    for paragraph in body.findall(f"{A_NS}p"):
        body.remove(paragraph)
    for line in lines:
        new_p = deepcopy(modelled)
        for run in new_p.findall(f"{A_NS}r"):
            new_p.remove(run)
        body.append(new_p)
        paragraph = frame.paragraphs[len(body.findall(f'{A_NS}p')) - 1]
        run = paragraph.add_run()
        run.text = line
        run.font.size = Pt(size)
        paragraph.space_before = Pt(6)
        properties = paragraph._p.get_or_add_pPr()
        properties.set("marL", "285750")
        properties.set("indent", "-285750")
        bullet = OxmlElement("a:buChar")
        bullet.set("char", "•")
        properties.append(bullet)


def add_table(slide, rows, left, top, width, height, widths, font=9.5):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    table = shape.table
    for index, value in enumerate(widths):
        table.columns[index].width = Inches(value)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.space_after = Pt(0)
            run = paragraph.add_run()
            run.text = str(value)
            run.font.size = Pt(font)
            run.font.bold = r == 0
            cell.margin_top = Emu(18000)
            cell.margin_bottom = Emu(18000)
            cell.margin_left = Emu(51000)
            cell.margin_right = Emu(51000)
    return table


def add_link(slide, text: str, url: str, left: float, top: float, size: float = 11) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(9.0), Inches(0.35))
    paragraph = box.text_frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0x0B, 0x5C, 0xAB)
    run.hyperlink.address = url


def reorder(prs, order: list[int]) -> None:
    sld_ids = prs.slides._sldIdLst
    elements = list(sld_ids)
    for element in elements:
        sld_ids.remove(element)
    for index in order:
        sld_ids.append(elements[index])


def main() -> int:
    prs = Presentation(str(DECK))
    if len(prs.slides) != 20:
        raise SystemExit(f"expected the 20-slide deck, found {len(prs.slides)}")

    # contents slide: state what each section holds, and link the repository
    set_body(prs.slides[1], OUTLINE, size=13.0)
    add_link(prs.slides[1], REPO, REPO, 0.54, 6.35)

    # slide 11: methodology as a table
    methodology = prs.slides[10]
    for shape in list(methodology.shapes):
        if not (shape.is_placeholder and shape.placeholder_format.idx == 0):
            shape._element.getparent().remove(shape._element)
    add_table(methodology, METHOD_ROWS, 0.42, 1.30, 9.95, 4.6, [1.85, 5.1, 3.0], font=10.5)

    # hyperparameter slide
    hyper = clone(prs, 15)   # a title+body slide, not the table-only methodology slide
    set_title(hyper, "Hyperparameters and training parameters")
    placeholder(hyper, 1)._element.getparent().remove(placeholder(hyper, 1)._element)
    add_table(hyper, HYPER_ROWS, 0.42, 1.20, 9.95, 5.2, [2.35, 2.62, 2.35, 2.63], font=9.0)

    # module-wise evidence slide
    evidence = clone(prs, 15)
    set_title(evidence, "Implementation evidence — module-wise results")
    placeholder(evidence, 1)._element.getparent().remove(placeholder(evidence, 1)._element)
    shots = [("mod1_data.png", 0.42, 1.12), ("mod2_train.png", 5.15, 1.12),
             ("mod3_latent.png", 0.42, 2.72), ("mod4_tests.png", 5.15, 2.72)]
    for name, left, top in shots:
        evidence.shapes.add_picture(str(ASSETS / name), Inches(left), Inches(top), width=Inches(4.55))
    caption = evidence.shapes.add_textbox(Inches(0.42), Inches(4.95), Inches(9.7), Inches(1.6))
    frame = caption.text_frame
    frame.word_wrap = True
    for index, line in enumerate(EVIDENCE_BULLETS):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(2)
        run = paragraph.add_run()
        run.text = "• " + line
        run.font.size = Pt(10.5)

    # demonstration slide: dashboard screenshot, bullets, repository link
    demo = prs.slides[14]
    set_title(demo, "Live demonstration — dashboard and repository")
    set_body(demo, DEMO_BULLETS, size=11.5)
    body = placeholder(demo, 1)
    body.width = Inches(5.2)
    body.height = Inches(3.4)
    demo.shapes.add_picture(str(ASSETS / "dashboard.png"), Inches(5.55), Inches(1.45),
                            width=Inches(4.85))
    add_link(demo, REPO, REPO, 0.54, 6.45, size=10.5)

    bullets = add_bullets(prs)

    # 20 = hyperparameters, 21 = module-wise evidence, both appended at the end
    order = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 11, 12, 21, 13, 14, 15, 16, 17, 18, 19])
    reorder(prs, order)

    prs.save(str(DECK))
    final = Presentation(str(DECK))
    print(f"bullets added: {bullets}; slides: {len(final.slides)}")
    for i, slide in enumerate(final.slides, 1):
        title = next((sh.text_frame.text.split("\n")[0] for sh in slide.shapes
                      if sh.is_placeholder and sh.placeholder_format.idx == 0), "—")
        print(f"  {i:2}. {title[:74]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
