"""Rebuild the interim deck to the mentor's slide-by-slide instructions.

Mentor's feedback: introduction on one slide, problem statement on its own slide, literature survey
as in the previous review, approved research gaps on one slide, objectives written out in full, and
dataset description / computing requirements under explicit sub-headings.

Strategy: keep the university template's master and layouts (so fonts, colours, footer and the
auto slide-number field stay intact), clone existing slides for the new pages, and reorder the
slide list at the end. Text is taken from the approved pre-project deck, not retyped from memory.
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
OUT = ROOT / "docs" / "Interim_Project_Presentation_7th-sem.pptx"

# --------------------------------------------------------------------------- approved content
INTRO = [
    "Text encoders turn words and sentences into vectors used for classification, search and ranking.",
    "Two ways to train them without labels: predict the hidden word (MLM, BERT), or predict a hidden "
    "representation (latent prediction, data2vec and JEPA).",
    "Latent prediction works in vision and for text at scale, but at small scale the evidence "
    "conflicts: one text recipe succeeds, another collapses [5, 8].",
    "Those recipes differ in target, masking, predictor and loss at the same time, so no study says "
    "which choice caused the difference (G1).",
    "Our question: under a fixed small encoder and dataset, which design choices keep masked latent "
    "prediction from collapsing and make its representations useful?",
]

PROBLEM_HEAD = ("The cause of collapse versus success is unknown, because every published recipe "
                "changes several things at once.")
PROBLEM = [
    "Masked latent prediction works for vision, but text evidence conflicts: success in data2vec [5], "
    "collapse in a 2026 text-JEPA study [8].",
    "A practitioner at academic scale therefore cannot tell whether a failure comes from the "
    "objective, the masking, the predictor, the regression loss, or missing regularisation.",
    "Research question: which design choice — teacher target, mask pattern, predictor, regression "
    "loss, or explicit collapse regularisation — decides whether masked latent prediction in a small "
    "text encoder collapses or learns transferable representations?",
    "Scope: one shared 4-layer encoder, identical data and token budget across conditions, free GPU "
    "only, every claim backed by a stored result.",
]

LIT = [
    {
        "title": "Literature survey — foundations of masked text representation learning [1]–[5]",
        "rows": [
            ["Ref.", "Title, year", "Methodology", "Dataset / evaluation", "Limitation for this study",
             "Observation"],
            ["[1]", "BERT, 2019", "Bidirectional encoder with MLM", "Books, Wikipedia; language tasks",
             "Predicts tokens, not teacher representations", "Establishes the MLM reference objective"],
            ["[2]", "RoBERTa, 2019", "Improved MLM training choices, dynamic masking",
             "Books + Wikipedia; GLUE", "Large original pretraining budget",
             "Training choices can change conclusions"],
            ["[3]", "ELECTRA, 2020", "Discriminates replaced tokens", "Text pretraining; GLUE",
             "Different target and generator setup", "Motivates efficient encoder comparisons"],
            ["[4]", "SimCSE, 2021", "Contrastive sentence representation learning",
             "Semantic similarity tasks", "Sentence-level objective, not masked latent regression",
             "Useful context for transfer evaluation"],
            ["[5]", "data2vec, 2022", "EMA teacher, masked student, contextual latent targets",
             "Text, vision, speech; GLUE", "Many recipe choices change together",
             "Shows masked text latent prediction can work"],
        ],
    },
    {
        "title": "Literature survey — the gap across modalities and objectives [6]–[10]",
        "rows": [
            ["Ref.", "Title, year", "Methodology", "Dataset / evaluation", "Limitation for this study",
             "Observation"],
            ["[6]", "data2vec 2.0, 2022", "More efficient contextual target prediction",
             "Text, vision, speech; GLUE", "Efficiency changes more than one component",
             "Confirms an efficient latent prediction path exists"],
            ["[7]", "I-JEPA, 2023", "Predicts masked image-region embeddings with an EMA target",
             "ImageNet and vision tasks", "Images differ from text",
             "Provides the JEPA architectural starting point"],
            ["[8]", "The JEPA Paradox in Language, 2026", "Analyses deterministic text JEPA and collapse",
             "Matched image/text; downstream transfer",
             "Does not isolate our choices on one small text encoder",
             "Reports collapse in its tested text setup"],
            ["[9]", "Predict and Reconstruct, 2026", "Hybrid JEPA and MLM encoder",
             "English Wikipedia; five GLUE tasks", "Hybrid objective does not isolate latent-only behaviour",
             "Geometry changed while probe accuracy stayed similar"],
            ["[10]", "ProteinJEPA, 2026", "Compares MLM, JEPA and their combination",
             "Protein sequences; biology tasks", "Protein sequences are a different domain",
             "JEPA alone often collapsed; hybrid results vary"],
        ],
    },
    {
        "title": "Literature survey — dynamics, evaluation tools and theory [11]–[15]",
        "rows": [
            ["Ref.", "Title, year", "Methodology", "Dataset / evaluation", "Limitation for this study",
             "Observation"],
            ["[11]", "LLM-JEPA, 2026", "Adds a JEPA embedding loss to the next-token loss",
             "NL-RX, GSM8K, Spider; decoder LLMs", "Decoder LLMs only; keeps the token loss",
             "Language-JEPA work is moving to large decoders"],
            ["[12]", "LeJEPA, 2025", "JEPA loss + SIGReg regulariser",
             "Vision joint-embedding SSL", "Not tested on masked text encoders",
             "Offers an anti-collapse remedy untested on text"],
            ["[13]", "RankMe, 2023", "Effective rank of embeddings as a label-free score",
             "Vision SSL benchmarks", "Validated on vision, not text encoders",
             "Links effective rank to downstream quality"],
            ["[14]", "How Contextual are Contextualized Word Representations?, 2019",
             "Layer-wise geometry analysis", "ELMo, BERT, GPT-2", "Analyses finished models, not objectives",
             "Text embeddings are anisotropic, so scale matters"],
            ["[15]", "Learn from Your Own Latents and Not from Tokens, 2026",
             "Sample-complexity theory of latent vs token prediction",
             "Synthetic hierarchical grammar", "Synthetic data only, no natural-language test",
             "Predicts the largest benefit with limited data"],
        ],
    },
]

GAPS = [
    ("G1 · Confounded comparisons",
     "data2vec-style latent prediction works on text; a JEPA-style text encoder collapses [8]. Those "
     "recipes differ in target, mask, predictor and loss all at once, so no study isolates the cause."),
    ("G2 · Latent-only encoders under-studied",
     "Language JEPA targets decoder LLMs and keeps the token loss [11]; encoders use MLM hybrids [9] "
     "or protein data [10]. There is no controlled latent-only study against an MLM control."),
    ("G3 · Collapse diagnosed, not remedied",
     "Normalised multi-layer targets [5] and SIGReg [12] prevent collapse outside text; [8] diagnoses "
     "it in text. These remedies have not been tested on masked text latent prediction."),
    ("G4 · Unreliable evaluation",
     "Loss hides quality [13], text is anisotropic [14], and geometry changed once without probe "
     "accuracy changing [9]. No protocol reads geometry and transfer together for text."),
    ("G5 · Small-budget regime untested",
     "Text latent-prediction evidence comes from large budgets [5], while theory predicts the largest "
     "benefit with limited data [15]. There is no measured evidence at academic scale on real text."),
]

OBJECTIVES = [
    "1. To analyse the text corpus and masking setup and find the token-frequency, sequence-length "
    "and masked-word ambiguity patterns that could cause latent prediction to collapse. (G1, G5)",
    "2. To develop a small encoder-only Transformer that learns by masked latent prediction from an "
    "EMA teacher, together with an MLM control model trained on the same data. (G2)",
    "3. To integrate collapse-prevention components (normalised multi-layer teacher targets, and "
    "SIGReg regularisation if time permits) into the latent-prediction model. (G3)",
    "4. To evaluate and validate the model's performance with each component switched on one at a "
    "time, using representation geometry, frozen downstream transfer and GPU-hours. (G1, G4, G5)",
]

DATASET = [
    ("Dataset description", "h"),
    "Corpus: WikiText-2 and WikiText-103 raw (Salesforce); 90,000 records per run, WikiText-103 "
    "used for the reported sweep.",
    "Tokenizer: bert-base-uncased used as a tokenizer only - no pretrained weights enter the model.",
    "Preprocessing: sequence length 128, 15% of positions masked, two policies (independent and "
    "span of 3), special tokens never selected for prediction.",
    "Evaluation task: AG News, 4,000 probe-train and 1,000 held-out examples; SST-2 held in reserve.",
    ("Computing requirements", "h"),
    "Free Kaggle T4 (15 GB, PyTorch 2.11 + cu128) inside a 30 GPU-hour weekly quota; no HPC cluster "
    "and no paid infrastructure.",
    "Measured: 24 training runs used 1.2 GPU-hours in total - 35 s for a 0.69M-token run, 353 s for "
    "a 5M-token run.",
    "Minimum viable hardware: one CUDA GPU with at least 8 GB. The pipeline is single-GPU, fully "
    "checkpointed, and needs no interconnects or fast storage.",
    "Local machine: CPU only, used for tests, analysis and figure generation.",
]

REFERENCES = [
    "[1] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT.",
    "[2] Liu, Y., Ott, M., Goyal, N., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692.",
    "[3] Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020). ELECTRA: Pre-training text encoders as discriminators rather than generators. ICLR.",
    "[4] Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple contrastive learning of sentence embeddings. EMNLP.",
    "[5] Baevski, A., Hsu, W.-N., Xu, Q., et al. (2022). data2vec: A general framework for self-supervised learning in speech, vision and language. ICML.",
    "[6] Baevski, A., Babu, A., Hsu, W.-N., & Auli, M. (2022). Efficient self-supervised learning with contextualized target representations for vision, speech and language. ICML.",
    "[7] Assran, M., Duquenne, P.-A., Misra, I., et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. CVPR.",
    "[8] The JEPA Paradox in Language (2026). Text-JEPA analysis and collapse diagnostics, preprint as cited in the approved proposal.",
    "[9] Predict and Reconstruct (2026). Hybrid JEPA and MLM encoder study, preprint as cited in the approved proposal.",
    "[10] ProteinJEPA (2026). Comparison of MLM, JEPA and hybrid objectives on protein sequences, preprint as cited in the approved proposal.",
    "[11] LLM-JEPA (2026). Adding joint-embedding prediction to decoder language models, ICLR.",
    "[12] LeJEPA (2025). JEPA loss with a SIGReg regulariser, preprint as cited in the approved proposal.",
    "[13] RankMe: Reliable human ratings for natural language generation (2023), ICML — effective-rank quality score.",
    "[14] Ethayarajh, K. (2019). How contextual are contextualized word representations? EMNLP-IJCNLP.",
    "[15] Learn from Your Own Latents and Not from Tokens (2026). Sample-complexity theory of latent prediction, preprint as cited in the approved proposal.",
    "[16] He, K., Chen, X., Xie, S., et al. (2022). Masked autoencoders are scalable vision learners. CVPR.",
    "[17] Grill, J.-B., Strub, F., Altché, F., et al. (2020). Bootstrap your own latent. NeurIPS.",
    "[18] Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-invariance-covariance regularization for self-supervised learning. ICLR.",
    "[19] Zbontar, J., Jing, L., Misra, I., et al. (2021). Barlow Twins: Self-supervised learning via redundancy reduction. ICML.",
    "[20] Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). Pointer sentinel mixture models (WikiText). ICLR.",
    "[21] Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification (AG News). NeurIPS.",
]

OUTLINE = [
    "Introduction",
    "Problem statement",
    "Literature survey and approved research gaps",
    "Approved objectives and objective-wise progress",
    "Methodology and system architecture",
    "Dataset description and computing requirements",
    "HPC lab requirement",
    "Implementation evidence and preliminary results",
    "Live demonstration",
    "Remaining work, timeline and risks",
    "SDG alignment and conclusion",
    "References",
]


# --------------------------------------------------------------------------- helpers
def clone(prs, model_index: int):
    """New slide carrying the same shapes as an existing one, so styling matches exactly."""
    model = prs.slides[model_index]
    layout = next(layout for layout in prs.slide_layouts if layout.name == "OBJECT")
    slide = prs.slides.add_slide(layout)
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)
    for shape in model.shapes:
        slide.shapes._spTree.append(deepcopy(shape._element))
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


def set_body(slide, lines, bold_lead: bool = False, size: float = 13.0) -> None:
    """Replace body text, reusing the model paragraph's own formatting.

    An entry is either a string, or a tuple (label, rest, style) where style is "h" for a
    sub-heading, "b" for a bold lead-in followed by normal text, or "n" for normal text.
    """
    frame = placeholder(slide, 1).text_frame
    frame.word_wrap = True
    modelled = deepcopy(frame._txBody.findall(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}p")[0])
    body = frame._txBody
    for paragraph in body.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}p"):
        body.remove(paragraph)
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    for line in lines:
        label, rest, style = None, line, "n"
        if isinstance(line, tuple):
            if len(line) == 3:
                label, rest, style = line
            else:
                label, style = line[0], line[1]
                rest = ""
        new_p = deepcopy(modelled)
        for run in new_p.findall(f"{ns}r"):
            new_p.remove(run)
        for extra in new_p.findall(f"{ns}br"):
            new_p.remove(extra)
        body.append(new_p)
        paragraph = frame.paragraphs[len(body.findall(f'{ns}p')) - 1]
        if style == "h":
            properties = paragraph._p.get_or_add_pPr()
            properties.set("marL", "0")
            properties.set("indent", "0")
            for bullet in properties.findall(
                    "{http://schemas.openxmlformats.org/drawingml/2006/main}buChar"):
                properties.remove(bullet)
            properties.append(OxmlElement("a:buNone"))
            run = paragraph.add_run()
            run.text = label
            run.font.bold = True
            run.font.size = Pt(15.5)
            run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x5C)
            paragraph.space_before = Pt(10)
            continue
        if not label and rest and rest[0].isdigit() and rest[1:3] == ". ":
            properties = paragraph._p.get_or_add_pPr()
            properties.set("marL", "0")
            properties.set("indent", "0")
            for bullet in properties.findall(
                    "{http://schemas.openxmlformats.org/drawingml/2006/main}buChar"):
                properties.remove(bullet)
            properties.append(OxmlElement("a:buNone"))
        if label:
            lead = paragraph.add_run()
            lead.text = label + " — "
            lead.font.bold = True
            lead.font.size = Pt(size)
        run = paragraph.add_run()
        run.text = rest
        run.font.size = Pt(size)
        paragraph.space_before = Pt(6 if style == "b" else 0)


def add_table(slide, rows, left=0.42, top=1.45, width=9.95, height=4.9, font=7.6):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    table = shape.table
    widths = [0.62, 1.62, 1.95, 1.85, 2.05, 1.86]
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
            cell.margin_top = Emu(20000)
            cell.margin_bottom = Emu(20000)
            cell.margin_left = Emu(45000)
            cell.margin_right = Emu(45000)
    return table


def add_references(slide) -> None:
    """Two balanced columns, because 21 entries will not fit in one body placeholder."""
    for shape in list(slide.shapes):
        if shape.is_placeholder and shape.placeholder_format.idx == 1:
            shape._element.getparent().remove(shape._element)
    half = (len(REFERENCES) + 1) // 2
    for index, chunk in enumerate((REFERENCES[:half], REFERENCES[half:])):
        box = slide.shapes.add_textbox(Inches(0.42 + index * 5.05), Inches(1.35),
                                       Inches(4.95), Inches(5.5))
        frame = box.text_frame
        frame.word_wrap = True
        for i, entry in enumerate(chunk):
            paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            paragraph.space_after = Pt(5)
            run = paragraph.add_run()
            run.text = entry
            run.font.size = Pt(7.8)


def reorder(prs, order: list[int]) -> None:
    """Rearrange slides by their 0-based index in the current presentation."""
    sld_ids = prs.slides._sldIdLst
    elements = list(sld_ids)
    for element in elements:
        sld_ids.remove(element)
    for index in order:
        sld_ids.append(elements[index])


# --------------------------------------------------------------------------- build
def main() -> int:
    prs = Presentation(str(DECK))
    if len(prs.slides) != 15:
        raise SystemExit(f"expected the 15-slide source deck, found {len(prs.slides)}; "
                         "restore it from git before rerunning")

    # rewrite in place: outline (1), introduction (2), dataset+compute (6), references (13)
    set_body(prs.slides[1], OUTLINE)
    set_title(prs.slides[2], "Introduction")
    set_body(prs.slides[2], INTRO)
    set_title(prs.slides[4], "Objective-wise progress against the approved objectives")
    progress = next(sh for sh in prs.slides[4].shapes if getattr(sh, "has_table", False)
                    and sh.has_table).table
    cell = progress.cell(3, 1)
    cell.text = ""
    run = cell.text_frame.paragraphs[0].add_run()
    run.text = ("4-layer encoder, MLM control, latent recipes A/B, EMA teacher, checkpoints. "
                "SIGReg was conditional and not run; VICReg and Barlow guards were implemented "
                "instead (Objective 4).")
    run.font.size = Pt(9)

    set_title(prs.slides[6], "Dataset and computing requirements")
    dataset_body = placeholder(prs.slides[6], 1)
    dataset_body.top = Inches(1.35)
    dataset_body.height = Inches(5.25)
    set_body(prs.slides[6], DATASET, size=12.5)

    set_title(prs.slides[13], "References")
    add_references(prs.slides[13])

    # new slides appended at the end, in the order we want them to appear
    problem = clone(prs, 10)
    set_title(problem, "Problem statement")
    set_body(problem, [PROBLEM_HEAD] + PROBLEM, size=13.5)
    problem_body = placeholder(problem, 1).text_frame
    problem_body.paragraphs[0].runs[0].font.italic = True

    lits = []
    for entry in LIT:
        slide = clone(prs, 10)
        set_title(slide, entry["title"])
        placeholder(slide, 1)._element.getparent().remove(placeholder(slide, 1)._element)
        add_table(slide, entry["rows"])
        lits.append(slide)

    gaps = clone(prs, 10)
    set_title(gaps, "Approved research gaps (G1–G5)")
    set_body(gaps, [(head, body, "b") for head, body in GAPS], size=13.0)

    objectives = clone(prs, 10)
    set_title(objectives, "Approved objectives")
    set_body(objectives, OBJECTIVES, size=13.5)

    new_indices = [15, 16, 17, 18, 19, 20]  # problem, lit x3, gaps, objectives
    order = (
        [0, 1, 2] + [new_indices[0]] + new_indices[1:4] + [new_indices[4], new_indices[5]]
        + [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    )
    # existing order is 0 title, 1 outline, 2 intro, 3 old literature, 4 objectives table,
    # 5 methodology, 6 dataset, 7 hpc, 8 results, 9 demo, 10 remaining, 11 sdg, 12 conclusion,
    # 13 references, 14 thanks. Slide 3 (old literature) is replaced by the dedicated slides.
    order = ([0, 1, 2, 15] + [16, 17, 18] + [19, 20] + [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    # drop the old combined literature slide (index 3)
    sld_ids = prs.slides._sldIdLst
    elements = list(sld_ids)
    for element in elements:
        sld_ids.remove(element)
    for index in order:
        sld_ids.append(elements[index])

    prs.save(str(OUT))
    final = Presentation(str(OUT))
    print(f"saved {OUT} with {len(final.slides)} slides")
    for i, slide in enumerate(final.slides, 1):
        title = next((sh.text_frame.text.split("\n")[0] for sh in slide.shapes
                      if sh.is_placeholder and sh.placeholder_format.idx == 0), "—")
        print(f"  {i:2}. {title[:78]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
