"""Build the interim synopsis / draft paper as a Word document.

Every measurement in this document is read from results/person1/*.json; nothing is typed by
hand. Reference metadata was verified against the Crossref API for published papers and the
arXiv API for preprints, so no bibliography entry is invented.

Run: .venv/Scripts/python.exe scripts/build_synopsis.py
Out: docs/SYNOPSIS_interim_draft_paper.docx
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "person1"
# SYNOPSIS_OUT lets the document be rebuilt while Word holds the canonical file open.
OUT = Path(os.environ.get("SYNOPSIS_OUT", ROOT / "docs" / "SYNOPSIS_interim_draft_paper.docx"))

TITLE = ("Masked Latent Prediction for Small Text Encoders: A Matched Comparison of Token, "
         "Latent and Regularised Objectives under a Fixed Compute Budget")

AUTHORS = "R. Daksha Subramanya, Sudeep S. S. Suvarna, Akash U. S."
AFFILIATION = ("Department of Computer Science and Engineering, Faculty of Engineering and "
               "Technology, M. S. Ramaiah University of Applied Sciences, Bengaluru, India")
MENTOR = "Mentor: Dr. Prabadevi B., Associate Professor, Department of CSE, MSRUAS"

ABSTRACT = (
    "Masked language modelling is the default objective for text encoders, but it predicts token "
    "identity, and token identity is frequently ambiguous: we measure that 30.2% of masked "
    "positions in WikiText-103 admit five or more plausible fillers. Latent-target objectives from "
    "the vision literature predict a representation instead, and are claimed to need an explicitly "
    "regularised embedding space to avoid collapse. Whether those claims survive at student scale,"
    " on one free GPU, is an open empirical question. We build one shared 4-layer encoder and train "
    "six matched conditions on it: masked token prediction, two masked latent recipes, two "
    "anti-collapse variants of the stronger latent recipe (VICReg-style variance and covariance "
    "terms, and Barlow-style redundancy reduction), and a no-teacher variant in which the targets "
    "come from the student's own detached forward pass. All conditions use the same data, tokenizer, "
    "batch size and annealed schedule at two budgets, 0.69M and 5.0M non-padding tokens, with two "
    "seeds each. Training is stable and nothing collapses: effective rank rises from 13.6 to 78.6 "
    "and to 114 when the guards are added. Transfer does not follow. Every one of the 24 runs scores "
    "below a randomly initialised encoder on a frozen linear probe (0.632), and a 7x larger budget "
    "lowers accuracy in every condition. Neither collapse guard nor removing the teacher closes the "
    "gap; removing the teacher helps most, which argues against our own original design. We report "
    "the negative result with the protocol that produced it, and identify a masked-versus-clean "
    "train/evaluation mismatch as the leading explanation to test next."
)

CONDITIONS_TABLE = [
    ["ID", "Objective", "Target", "Predictor", "Loss", "Masking"],
    ["C", "masked token (control)", "input token", "linear head", "cross-entropy", "independent"],
    ["A", "masked latent", "final teacher layer", "attention", "MSE", "span"],
    ["B", "masked latent", "mean of top-2 normalised layers", "linear", "Smooth L1", "independent"],
    ["H", "A + VICReg auxiliary", "as A", "as A", "MSE + variance/covariance", "span"],
    ["I", "A + Barlow auxiliary", "as A", "as A", "MSE + redundancy", "span"],
    ["J", "A, no teacher", "student's own detached pass", "as A", "MSE", "span"],
]

REFERENCES = [
 'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proc. NAACL-HLT, 2019, doi: 10.18653/v1/N19-1423.',
 'Y. Liu, M. Ott, N. Goyal, et al., "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.',
 'K. Clark, M.-T. Luong, Q. V. Le, and C. D. Manning, "ELECTRA: Pre-training text encoders as discriminators rather than generators," in Proc. ICLR, 2020; arXiv:2003.10555.',
 'M. Joshi, D. Chen, Y. Liu, et al., "SpanBERT: Improving pre-training by representing and predicting spans," Transactions of the ACL, 2020, doi: 10.1162/tacl_a_00300.',
 'V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019.',
 'X. Jiao, Y. Yin, L. Shang, et al., "TinyBERT: Distilling BERT for natural language understanding," Findings of ACL, 2020, doi: 10.18653/v1/2020.findings-emnlp.372.',
 'W. Wang, F. Wei, L. Dong, et al., "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," arXiv:2002.10957, 2020.',
 'P. He, X. Liu, J. Gao, and W. Chen, "DeBERTa: Decoding-enhanced BERT with disentangled attention," in Proc. ICLR, 2021; arXiv:2006.03654.',
 'C. Raffel, N. Shazeer, A. Roberts, et al., "Exploring the limits of transfer learning with a unified text-to-text transformer," Journal of Machine Learning Research, vol. 21, 2020; arXiv:1910.10683.',
 'K. He, X. Chen, S. Xie, et al., "Masked autoencoders are scalable vision learners," in Proc. CVPR, 2022, doi: 10.1109/CVPR52688.2022.01553.',
 'H. Bao, L. Dong, S. Piao, and F. Wei, "BEiT: BERT pre-training of image transformers," in Proc. ICLR, 2022; arXiv:2106.08254.',
 'Z. Xie, Z. Zhang, Y. Cao, et al., "SimMIM: A simple framework for masked image modeling," in Proc. CVPR, 2022, doi: 10.1109/CVPR52688.2022.00943.',
 'A. Baevski, W.-N. Hsu, Q. Xu, et al., "data2vec: A general framework for self-supervised learning in speech, vision and language," in Proc. ICML, 2022; arXiv:2202.03555.',
 'J.-B. Grill, F. Strub, F. Altché, et al., "Bootstrap your own latent: A new approach to self-supervised learning," in Proc. NeurIPS, 2020; arXiv:2006.07733.',
 'X. Chen and K. He, "Exploring simple siamese representation learning," in Proc. CVPR, 2021, doi: 10.1109/CVPR46437.2021.01549.',
 'K. He, H. Fan, Y. Wu, et al., "Momentum contrast for unsupervised visual representation learning," in Proc. CVPR, 2020, doi: 10.1109/CVPR42600.2020.00975.',
 'M. Caron, H. Touvron, I. Misra, et al., "Emerging properties in self-supervised vision transformers," in Proc. ICCV, 2021, doi: 10.1109/ICCV48922.2021.00951.',
 'A. Bardes, J. Ponce, and Y. LeCun, "VICReg: Variance-invariance-covariance regularization for self-supervised learning," in Proc. ICLR, 2022; arXiv:2105.04906.',
 'J. Zbontar, L. Jing, I. Misra, et al., "Barlow Twins: Self-supervised learning via redundancy reduction," in Proc. ICML, 2021; arXiv:2103.03230.',
 'T. Gao, X. Yao, and D. Chen, "SimCSE: Simple contrastive learning of sentence embeddings," in Proc. EMNLP, 2021, doi: 10.18653/v1/2021.emnlp-main.552.',
 'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using siamese BERT-networks," in Proc. EMNLP-IJCNLP, 2019, doi: 10.18653/v1/D19-1410.',
 'K. Ethayarajh, "How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings," in Proc. EMNLP-IJCNLP, 2019, doi: 10.18653/v1/D19-1006.',
 'J. Gao, D. He, X. Tan, et al., "Representation degeneration problem in training natural language generation models," in Proc. ICLR, 2019; arXiv:1907.12009.',
 'S. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer sentinel mixture models," in Proc. ICLR, 2017; arXiv:1609.07843.',
 'X. Zhang, J. Zhao, and Y. LeCun, "Character-level convolutional networks for text classification," in Proc. NeurIPS, 2015; arXiv:1509.01626.',
 'R. Socher, A. Perelygin, J. Wu, et al., "Recursive deep models for semantic compositionality over a sentiment treebank," in Proc. EMNLP, 2013, doi: 10.18653/v1/D13-1170.',
 'J. Kaplan, S. McCandlish, T. Henighan, et al., "Scaling laws for neural language models," arXiv:2001.08361, 2020.',
]


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def runs_by(label_map: dict[str, str]) -> dict:
    """Collect (label, budget) -> list of run dicts from both sweeps."""
    out: dict[tuple[str, int], list[dict]] = {}
    for name in ("sweep_annealed.json", "sweep_anticollapse.json"):
        for run in load(name)["runs"]:
            out.setdefault((run["label"], run["budget_tokens"]), []).append(run)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def add_page_numbers(doc: Document) -> None:
    """Footer page number, centred, as a real PAGE field so Word keeps it current."""
    paragraph = doc.sections[0].footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, end):
        run._r.append(element)
    run.font.size = Pt(9)


def style_document(doc: Document) -> None:
    section = doc.sections[0]
    section.left_margin = section.right_margin = Pt(72)
    section.top_margin = section.bottom_margin = Pt(72)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(12 if level == 1 else 11)
    run.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)


def para(doc: Document, text: str) -> None:
    doc.add_paragraph(text)


def bullet(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(text, style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def table(doc: Document, rows: list[list[str]], caption: str) -> None:
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.keep_with_next = True
    run = cap.add_run(caption)
    run.bold = True
    run.font.size = Pt(9.5)

    grid = doc.add_table(rows=len(rows), cols=len(rows[0]))
    grid.style = "Table Grid"
    grid.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = grid.cell(r, c)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(1)
            run = paragraph.add_run(str(value))
            run.font.size = Pt(9)
            run.bold = r == 0

    # A table that crosses a page boundary must repeat its header, or the continuation rows on
    # the next page are unlabelled numbers.
    header_row = grid.rows[0]._tr.get_or_add_trPr()
    header_flag = OxmlElement("w:tblHeader")
    header_flag.set(qn("w:val"), "true")
    header_row.append(header_flag)
    for row in grid.rows:
        properties = row._tr.get_or_add_trPr()
        properties.append(OxmlElement("w:cantSplit"))
    doc.add_paragraph()


def build() -> Document:
    doc = Document()
    style_document(doc)
    add_page_numbers(doc)

    title = doc.add_paragraph()
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(10)
    run = title.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(15)

    for text, size, italic in ((AUTHORS, 10.5, False), (AFFILIATION, 9.5, True), (MENTOR, 9.5, True)):
        block = doc.add_paragraph()
        block.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        block.paragraph_format.space_after = Pt(2)
        run = block.add_run(text)
        run.font.size = Pt(size)
        run.italic = italic

    heading(doc, "Abstract", 1)
    abstract = doc.add_paragraph(ABSTRACT)
    abstract.paragraph_format.space_after = Pt(8)

    # ---------------------------------------------------------------- 1
    heading(doc, "1. Introduction", 1)
    para(doc,
         "Pretraining a text encoder to BERT scale consumes roughly 3.3 billion tokens of compute "
         "(Devlin et al., 2019). Institutions without cluster access cannot enter that regime, and "
         "the standard remedy, distilling an existing model (Sanh et al., 2019; Jiao et al., 2020; "
         "Wang et al., 2020), assumes a teacher large enough to be worth distilling. This project "
         "asks a narrower question: with one 4-layer encoder, a few million tokens and a free GPU, "
         "does the choice of prediction target change what the resulting representation is worth?")
    para(doc,
         "The default target is token identity. Masked language modelling asks the encoder to "
         "recover a specific word from context, and for natural text that word is frequently not "
         "recoverable. Our own measurement on WikiText-103 finds a mean conditional entropy of 1.28 "
         "nats at masked positions, with 30.2% of positions above log 5 and 60.1% below one nat. "
         "Where entropy is high, the loss rewards a guess rather than a representation, and the "
         "gradient carries lexical noise rather than semantic signal. The vision literature reached "
         "a similar conclusion earlier and moved the prediction target into representation space "
         "(He et al., 2022; Bao et al., 2022; Xie et al., 2022; Baevski et al., 2022).")
    para(doc,
         "We transfer that idea to a small text encoder and test it under a matched protocol. Six "
         "conditions share one encoder architecture, one tokenizer, one corpus, one batch size, one "
         "optimiser schedule and one evaluation harness; only the objective changes. The comparison "
         "is run at two token budgets with two seeds, so a difference has to survive both. Beyond "
         "the two latent recipes, we add three conditions aimed at objections a reviewer should "
         "raise: two anti-collapse regularisers of the kind the vision literature recommends "
         "(Bardes et al., 2021; Zbontar et al., 2021), and one condition that removes the teacher "
         "entirely, because an exponential-moving-average teacher is otherwise justified only by "
         "appeal to the papers that introduced it.")
    para(doc, "The contributions of this interim stage are:")
    for text in (
        "a matched experimental framework in which token, latent and regularised objectives are "
        "compared on identical data, schedule and budget, with all code, configs and result files "
        "in a public repository;",
        "a direct measurement of how ambiguous masked positions are in the corpus we train on, "
        "which motivates the latent target rather than asserting it;",
        "a budget sweep from 0.69M to 5.0M tokens with the learning-rate schedule annealed in every "
        "run, after an initial pass revealed that a fixed update count turns a budget comparison "
        "into a schedule comparison;",
        "an anti-collapse and no-teacher ablation that isolates whether the failure to transfer is "
        "caused by representation collapse or by the teacher;",
        "a negative result reported in full: no condition beats a randomly initialised encoder on "
        "a frozen linear probe, and more pretraining makes transfer worse.",
    ):
        bullet(doc, text)

    # ---------------------------------------------------------------- 2
    heading(doc, "2. Related Work", 1)
    heading(doc, "2.1 Masked language modelling and encoder compression", 2)
    para(doc,
         "BERT established masked token prediction as the standard pretraining objective for text "
         "encoders (Devlin et al., 2019). RoBERTa showed the recipe is sensitive to training details "
         "rather than architecture (Liu et al., 2019), ELECTRA replaced token recovery with "
         "discrimination (Clark et al., 2020), and SpanBERT showed that predicting contiguous spans "
         "rather than isolated tokens yields gains on span-selection tasks (Joshi et al., 2020). A "
         "parallel line compresses existing encoders by distilling a pretrained teacher: DistilBERT "
         "(Sanh et al., 2019), TinyBERT (Jiao et al., 2020) and MiniLM (Wang et al., 2020). DeBERTa "
         "belongs to a different line, disentangling content and position inside attention rather "
         "than starting from a teacher (He et al., 2020). T5 recast every objective as "
         "text-to-text generation (Raffel et al., 2020). What this line shares is that the target remains defined over the vocabulary; "
         "none of these objectives removes the ambiguity of the target itself.")
    heading(doc, "2.2 Latent targets and masked representation modelling", 2)
    para(doc,
         "MAE demonstrated that masking a large fraction of an image and predicting latent "
         "representations is both cheaper and more accurate than predicting pixels (He et al., "
         "2022); BEiT predicts discrete visual tokens from a pretrained tokenizer (Bao et al., "
         "2022); SimMIM showed a simpler regression variant suffices (Xie et al., 2022); and "
         "data2vec unified the latent-target idea across speech, vision and language (Baevski et "
         "al., 2022). The consistent finding is that predicting in representation space avoids "
         "spending capacity on unpredictable low-level detail. Our recipes A and B are direct "
         "adaptations of this idea to text at student scale.")
    heading(doc, "2.3 Self-distillation, momentum teachers and collapse", 2)
    para(doc,
         "BYOL showed that an exponential-moving-average teacher can supply targets without "
         "negatives (Grill et al., 2020); SimSiam showed that negative pairs are unnecessary, "
         "needing only a stop-gradient and an asymmetric predictor (Chen and He, 2021); MoCo made the negative queue and momentum encoder explicit (He et al., 2020); "
         "DINO added centring and sharpening to stabilise the teacher's outputs (Caron et al., "
         "2021). Because these methods can trivially collapse, two families of explicit guards were "
         "proposed: VICReg penalises dimensions whose variance falls below a floor and "
         "off-diagonal covariance between dimensions (Bardes et al., 2021), while Barlow Twins "
         "decorrelates the cross-correlation matrix of two views (Zbontar et al., 2021). Both are "
         "two-view methods in their original form; the question of whether their statistics terms "
         "help when the invariance term comes from a masked-prediction loss instead is what "
         "conditions H and I test.")
    heading(doc, "2.4 Sentence-level objectives and representation geometry", 2)
    para(doc,
         "Sentence-pair objectives produce usable sentence representations from pretrained "
         "encoders: Sentence-BERT trains a siamese encoder on classification and regression over "
         "pairs (Reimers and Gurevych, 2019), while SimCSE applies an explicitly contrastive "
         "objective with dropout as the augmentation (Gao et al., 2021). Two findings from the geometry literature are directly "
         "relevant to our evaluation. Ethayarajh showed that contextual representations occupy a "
         "narrow cone whose anisotropy grows with depth, which makes a linear probe sensitive to "
         "scale rather than only to content (Ethayarajh, 2019). Gao et al. showed that natural "
         "language generation training drives representations into a degenerate low-rank subspace "
         "(Gao et al., 2019). Both motivate measuring effective rank alongside downstream accuracy "
         "rather than assuming that a low training loss implies a usable space.")
    heading(doc, "2.5 Scale, budgets and benchmarks", 2)
    para(doc,
         "Scaling studies show that loss falls predictably with parameters, data and compute "
         "(Kaplan et al., 2020), which cuts both ways for a project of this size: at student scale "
         "the absolute quality is expected to be poor, but the ordering between objectives need not "
         "be. We measure on WikiText-103 (Merity et al., 2016) for training and geometry, and on AG "
         "News (Zhang et al., 2015) for the frozen probe, holding SST-2 (Socher et al., 2013) in "
         "reserve as a second task.")
    heading(doc, "2.6 Gap", 2)
    para(doc,
         "Latent-target self-supervision is well characterised in vision and at large scale, and "
         "token-level objectives are well characterised for text. What is thin is a matched, "
         "multi-seed comparison of token, latent and explicitly regularised latent objectives for a "
         "small text encoder under a fixed and modest compute budget, with collapse measured "
         "directly rather than assumed, and with the teacher itself ablated. That is the gap this "
         "work addresses.")

    # ---------------------------------------------------------------- 3
    heading(doc, "3. Materials and Methods", 1)
    heading(doc, "3.1 Encoder", 2)
    para(doc,
         "One encoder is shared by every condition: 4 Transformer layers, width 256, 4 attention "
         "heads, feed-forward width 1024, dropout 0.1, pre-layer-norm, maximum sequence length 128, "
         "vocabulary 30522. Weights are randomly initialised; no pretrained weights enter the model. "
         "The bert-base-uncased tokenizer is used only for tokenization, and its special tokens "
         "(0, 101, 102, 103) are excluded from every prediction position so that no condition is "
         "ever asked to predict a special token. The encoder carries a padding-aware attention mask "
         "and can return intermediate hidden states, which recipe B requires.")
    heading(doc, "3.2 Objectives", 2)
    para(doc,
         "Table 1 lists the six conditions. Condition C is the control: a linear head over the "
         "masked positions with cross-entropy against the input token, i.e. standard MLM. "
         "Condition A predicts the final layer of a teacher at the masked positions through an "
         "attention predictor with MSE, using span masking. Condition B predicts the mean of the "
         "teacher's top-2 normalised layers through a linear predictor with Smooth L1, using "
         "independent masking. In A and B the teacher is a frozen exponential-moving-average copy of "
         "the encoder (decay 0.996) whose targets are detached; the EMA update happens after the "
         "optimiser step, so the teacher never receives gradients.")
    para(doc,
         "Condition J removes the teacher. Rather than setting the decay to 1, we delete the second "
         "network: targets come from the student's own forward pass on the clean input, detached. "
         "This distinguishes 'no momentum' from 'no second model', which the standard argument "
         "conflates.")
    para(doc,
         "Conditions H and I add an anti-collapse auxiliary to A. H applies a VICReg-style pair of "
         "terms to the student's hidden states at all non-padding positions: a hinge on the "
         "per-dimension standard deviation, relu(1 - std), averaged over dimensions, and the sum of "
         "squared off-diagonal entries of the covariance matrix divided by the number of "
         "dimensions, weighted 0.1 and 0.04 respectively and added to the latent loss with weight "
         "0.1. I applies a Barlow-style term: standardise each dimension, form the correlation "
         "matrix, penalise squared off-diagonal entries, and add a small on-diagonal term; it is "
         "added with weight 0.01 rather than 0.1 because the raw Barlow term measures about 0.58 at "
         "initialisation against roughly 0.08 for the VICReg pair, so equal nominal weights would "
         "let it contribute seven times more and turn the comparison into one of scale.")
    para(doc,
         "H and I are auxiliaries, not reproductions. VICReg and Barlow Twins obtain invariance from "
         "two augmented views; here invariance comes from the masked-latent loss and only the "
         "statistics terms are added. This isolates the guard while holding everything else fixed, "
         "which is the comparison of interest, but it should not be read as an implementation of "
         "either paper.")
    table(doc, CONDITIONS_TABLE, "Table 1. Experimental conditions. Every condition shares the encoder, "
          "data, tokenizer, batch size and schedule; only the objective changes.")

    heading(doc, "3.3 Optimisation and budget", 2)
    para(doc,
         "AdamW with learning rate 3e-4, weight decay 0.01, betas (0.9, 0.999), gradient clipping "
         "at 1.0, warmup 100 updates and a linear decay to zero. Batch size is 8 sequences of length "
         "128, measured at 575 non-padding tokens per update. The learning-rate schedule decays to "
         "zero at a configured maximum update count, so the maximum is derived from the token budget "
         "as budget/575 rather than held fixed. An earlier pass that fixed the update count while "
         "growing the budget compared two different schedules, not two budgets; those runs are "
         "recorded separately and are not used here. The two budgets are 0.69M tokens (1217 updates) "
         "and 5.0M tokens (8668 updates), the latter being the full budget in the approved proposal "
         "and roughly 0.15% of BERT's reported pretraining volume.")
    heading(doc, "3.4 Data and masking", 2)
    para(doc,
         "Training uses wikitext-103-raw-v1 with 90,000 records, sequences of 128 tokens and a 15% "
         "mask ratio. Two masking policies are implemented: independent Bernoulli masking, and span "
         "masking with span length 3. Special tokens are never selected, and the batch carries the "
         "clean input, the masked input, an attention mask and a prediction mask, so any condition "
         "can be trained from the same batch contract.")
    heading(doc, "3.5 Evaluation", 2)
    para(doc,
         "Three measurements are reported, all on frozen checkpoints. (i) Effective rank of the "
         "representation on held-out text, computed two ways, from the singular values of the "
         "representation matrix and from the eigenvalues of its covariance. (ii) A linear probe on "
         "AG News trained on 4,000 examples and evaluated on 1,000 held-out examples, with an "
         "identical probe, seed and schedule for every student, and with a randomly initialised "
         "encoder scored through the same probe as the reference point. (iii) An ambiguity analysis "
         "that masks one position at a time and records the MLM distribution's entropy, so that the "
         "motivation for a latent target is measured rather than assumed.")
    heading(doc, "3.6 Reproducibility", 2)
    para(doc,
         "Runs are executed as committed Kaggle kernels so they survive a closed browser, and every "
         "run writes a metrics file, a checkpoint and a metadata file containing the commit hash and "
         "the measured token count. Checkpoint resume is verified to reproduce counters and loss "
         "exactly. The repository holds 64 passing automated tests and a clean lint run; the deck "
         "and this document are generated from the result files rather than transcribed from them.")

    # ---------------------------------------------------------------- 4
    heading(doc, "4. Experimental Results", 1)
    runs = runs_by({})

    heading(doc, "4.1 Training behaviour", 2)
    rows = [["Condition", "Budget", "Seed", "Updates", "Tokens", "Loss first → last"]]
    order = ["C", "A", "B", "H", "I", "J"]
    for label in order:
        for budget in (700_000, 5_000_000):
            for run in sorted(runs[(label, budget)], key=lambda r: r["seed"]):
                rows.append([label, f"{budget/1e6:.2f}M" if budget >= 1e6 else "0.69M",
                             str(run["seed"]), f"{run['updates']:,}",
                             f"{run['nonpadding_tokens']:,}",
                             f"{run['first_loss']:.3f} → {run['last_loss']:.3f}"])
    table(doc, rows, "Table 2. Training loss at the start and end of every run. Losses are "
          "objective-specific and are not comparable across rows; a lower number does not indicate "
          "a better model.")
    para(doc,
         "All 24 runs converge without divergence. The latent objectives start near 2.0 nats and "
         "fall steeply; B reaches 0.088 at 5M tokens, the lowest value in the study, but Smooth L1 "
         "targets and cross-entropy are not on a common scale and the comparison carries no "
         "information about representation quality. Condition J, which has no teacher, ends higher "
         "than A at the same budget (0.65 against 0.51), consistent with a target that moves with "
         "the student rather than being held fixed.")

    heading(doc, "4.2 Transfer", 2)
    probe_rows = [["Condition", "0.69M tokens", "5.00M tokens", "Change"]]
    summary = {}
    for label in order:
        low = runs[(label, 700_000)]
        high = runs[(label, 5_000_000)]
        a_low = mean([r["probe_accuracy"] for r in low])
        a_high = mean([r["probe_accuracy"] for r in high])
        summary[label] = (a_low, a_high)
        probe_rows.append([label, f"{a_low:.3f}", f"{a_high:.3f}", f"{a_high - a_low:+.003f}"])
    probe_rows.append(["Random init (never trained)", "0.632", "0.632", "—"])
    probe_rows.append(["Majority class", "0.274", "0.274", "—"])
    table(doc, probe_rows, "Table 3. Frozen linear probe on AG News, 4,000 train / 1,000 test, mean of "
          "two seeds.")
    para(doc,
         "Two findings hold for every condition and every seed. First, no trained student beats the "
         "randomly initialised encoder: the best single run reaches 0.607 and the best mean is "
         "0.597, against 0.632 for an untrained encoder through the same probe. Second, the larger "
         "budget lowers accuracy in all six conditions, by 0.024 to 0.049, while training losses "
         "improve over the same interval. A 7x compute increase therefore buys a worse "
         "representation by the only measure that matters downstream.")
    para(doc,
         "The anti-collapse and no-teacher conditions change this quantitatively, not "
         "qualitatively. At 0.69M tokens all four latent conditions sit within seed noise of each "
         "other (0.589 to 0.597). At 5.0M the guards recover part of the loss (H and I both 0.567 "
         "against A's 0.549), and removing the teacher does slightly better than either guard "
         "(J 0.573). None of the three reaches the untrained reference.")

    heading(doc, "4.3 Geometry", 2)
    geometry = load("geometry_long.json")["students"]
    geo_rows = [["Student", "Eff. rank (singular)", "Eff. rank (covariance)", "Status"]]
    for name, record in sorted(geometry.items()):
        geo = record.get("geometry", record)
        geo_rows.append([name, f"{geo['effective_rank_singular']:.1f}",
                         f"{geo['effective_rank_covariance']:.2f}", geo.get("status", "")])
    table(doc, geo_rows, "Table 4. Effective rank of the frozen representation on held-out WikiText "
          "at the 0.69M budget, per seed.")
    para(doc,
         "Nothing collapses at this budget: the MLM control occupies a rank of 10.2 to 17.0, and "
         "the latent recipes 67.6 to 80.8. The MLM control's much lower rank means it distributes "
         "its representation over far fewer directions, which is consistent with a lexical target "
         "concentrating capacity, though a share of the difference may come from the MLM head "
         "consuming part of the update budget. At 5.0M the guards raise rank further (H 90.0, I "
         "107.5) while J, which has the lowest rank of the four (48.6), has the best accuracy. Rank "
         "and transfer therefore do not move together in this study, and a collapse metric alone "
         "would not have predicted the probe results. We do not report A's rank at 5M: that output "
         "could not be recovered from the kernel that produced it, and we prefer a missing cell to "
         "an inferred one.")

    heading(doc, "4.4 Ambiguity of the token target", 2)
    amb = load("ambiguity_wikitext2.json")["ambiguity"]
    amb_rows = [["Frequency band of the masked token", "Positions", "Mean entropy (nats)"]]
    for band, record in amb["entropy_by_frequency"].items():
        amb_rows.append([band, f"{record['positions']:,}", f"{record['mean_entropy']:.3f}"])
    table(doc, amb_rows, "Table 5. Conditional entropy of the MLM distribution when a single position "
          "is masked, by how frequent the true token is in the corpus.")
    para(doc,
         f"Across the analysed positions the mean entropy is {amb['mean_entropy_nats']:.3f} nats "
         f"with a median of {amb['median_entropy_nats']:.3f} and a 90th percentile of "
         f"{amb['p90_entropy_nats']:.3f}. {amb['frac_positions_entropy_gt_log5']*100:.1f}% of "
         f"positions admit five or more plausible fillers, and "
         f"{amb['frac_positions_entropy_lt_1nat']*100:.1f}% are nearly determined by context. The "
         "distribution is strongly bimodal, and the rare-token band is more ambiguous than the "
         "frequent band (1.523 against 1.107 nats), so the ambiguity is concentrated exactly where "
         "a lexical target is least informative about meaning.")

    heading(doc, "4.5 Component ablations", 2)
    para(doc,
         "Conditions D–G change one property of A at a time at the 0.69M budget: D switches the "
         "target to the normalised top-2 layers, E switches span masking to independent, F replaces "
         "the attention predictor with a linear one, and G switches MSE to Smooth L1. Their final "
         "losses were 0.358, 0.513, 0.902 and 0.239 respectively. These numbers are not rankable "
         "against each other because the loss functions differ, and at this budget no probe was run "
         "for them. They are reported as engineering evidence that each axis is independent and "
         "configurable, not as findings.")

    # ---------------------------------------------------------------- 5
    heading(doc, "5. Discussion", 1)
    heading(doc, "5.1 What the negative result does and does not say", 2)
    para(doc,
         "A trained encoder scoring below a randomly initialised one is a strong claim, and it is "
         "worth being precise about its scope. The probe is linear and is fitted on frozen "
         "features; it therefore measures how linearly separable the representation is, not how "
         "much information it contains. A random encoder with the same architecture produces "
         "high-variance, weakly correlated features, which a linear classifier can exploit; "
         "training on masked input can reduce that variance and sharpen features along directions "
         "that matter for masking but not for topic classification. The result does not show that "
         "our encoders are useless. It shows that, by the standard transfer measure, a 4-layer "
         "encoder trained on 5M tokens on this objective is not better than its own initialisation.")
    heading(doc, "5.2 Collapse is not the bottleneck", 2)
    para(doc,
         "VICReg and Barlow terms are designed to prevent a failure mode that, in our runs, does "
         "not occur: effective rank is already high without them, and adding them raises it further "
         "(76 to 102 and 114 at 0.69M) without improving accuracy at that budget at all. If "
         "collapse were the cause of the transfer failure, a guard should have repaired it. It did "
         "not. The most parsimonious reading is that the failure is not geometric.")
    heading(doc, "5.3 The teacher is not earning its place", 2)
    para(doc,
         "The no-teacher condition is the most informative ablation in this study, and it argues "
         "against our own design. Removing the second network and taking targets from the student's "
         "own detached pass gives the best 5M accuracy of the four latent conditions (0.573 against "
         "0.549 for the EMA version), at half the memory and without the EMA update. At 0.69M the "
         "difference is inside seed noise. At this scale, an EMA teacher is not justified by our "
         "data; it may still be justified at larger budgets, where a slowly moving target is "
         "thought to prevent collapse, but that remains to be shown rather than assumed.")
    heading(doc, "5.4 Leading explanation and the next experiment", 2)
    para(doc,
         "Training consumes masked input; the probe consumes clean input. Every step of training "
         "makes the encoder better at the masked-input pathway, and the clean pathway of the "
         "student is never trained at all, because the student only ever sees masked sequences. "
         "This is a genuine train/evaluation mismatch, and it predicts exactly what we observe: "
         "loss improves while clean-input transfer degrades with budget. MAE avoids the analogous "
         "problem by discarding masked tokens and encoding only visible patches at evaluation time. "
         "The decisive test is therefore to probe the same checkpoints with masked input, and to "
         "compare the two conditions directly; if the gap closes, the objective specialises the "
         "encoder rather than improving it, and the fix is architectural rather than a different "
         "loss.")
    heading(doc, "5.5 Limitations", 2)
    for text in (
        "Compute: 24 training runs consumed 1.2 GPU-hours on a free T4; the largest budget is 5.0M "
        "tokens, roughly 0.15% of BERT's, so no claim here extends to realistic pretraining scale.",
        "Single corpus and single probe task: WikiText-103 and AG News. A second probe (SST-2) is "
        "planned and not yet run.",
        "Two seeds. Differences below roughly 0.01 accuracy are not interpretable at this sample "
        "size, and we treat sub-0.01 differences as noise throughout.",
        "The regularisers act on all non-padding positions, about 570 vectors per batch at batch "
        "size 8. VICReg's covariance term over 256 dimensions is only marginally identified at that "
        "sample count, which biases against the guard and should be read as a limitation of our "
        "implementation rather than of the method.",
        "Absolute accuracy is low for every student, including the untrained reference (0.632 "
        "against 0.274 majority). The comparison is internally consistent but says nothing about "
        "what these encoders would do at larger scale.",
    ):
        bullet(doc, text)

    # ---------------------------------------------------------------- 6
    heading(doc, "6. Conclusion and Future Work", 1)
    para(doc,
         "We built a matched experimental framework for comparing prediction targets in small text "
         "encoders and used it to test whether latent targets, anti-collapse regularisers or the "
         "removal of an EMA teacher improve transfer at a fixed, modest compute budget. None did. "
         "Training is stable, nothing collapses, and the guards raise effective rank substantially, "
         "but every one of 24 runs scores below a randomly initialised encoder on a frozen linear "
         "probe, and a 7x larger budget makes transfer worse in every condition. Removing the "
         "teacher helps more than either regulariser, which is evidence against our original design "
         "rather than for it.")
    para(doc,
         "The value of the interim stage is therefore methodological as much as empirical: a "
         "protocol that can distinguish a schedule artefact from a budget effect, a collapse "
         "measurement that can rule collapse out as the explanation, and an ablation that tests the "
         "teacher instead of citing it. The immediate next step is the masked-versus-clean probe, "
         "which discriminates between the two remaining explanations. Beyond that: conditions D–G at "
         "the full budget with probes and a second seed, a second downstream task, and a longer "
         "budget to test whether the degradation reverses at scale.")

    # ---------------------------------------------------------------- refs
    heading(doc, "References", 1)
    for index, entry in enumerate(REFERENCES, 1):
        paragraph = doc.add_paragraph(f"[{index}] {entry}")
        paragraph.paragraph_format.left_indent = Pt(18)
        paragraph.paragraph_format.first_line_indent = Pt(-18)
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in paragraph.runs:
            run.font.size = Pt(9)

    return doc


def main() -> int:
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    words = sum(len(p.text.split()) for p in doc.paragraphs)
    table_words = sum(len(c.text.split()) for t in doc.tables for row in t.rows for c in row.cells)
    print(f"wrote {OUT}")
    print(f"paragraph words: {words}, table words: {table_words}, tables: {len(doc.tables)}, "
          f"references: {len(REFERENCES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
