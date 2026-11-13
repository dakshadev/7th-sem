"""Write the content payload that fill_deck_slides.ps1 feeds into PowerPoint.

Keeping the text here means the deck content stays reviewable and versioned, while the actual
editing happens inside PowerPoint, which cannot produce the corrupt packages python-pptx was
writing when slides were added to this particular deck.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extend_deck_for_review import EVIDENCE_BULLETS, HYPER_ROWS, METHOD_ROWS  # noqa: E402

OUT = ROOT / "docs" / "assets" / "deck_payload.json"


def main() -> int:
    payload = {
        "methodology_title": "Methodology — five modules",
        "methodology_rows": [[str(cell) for cell in row] for row in METHOD_ROWS],
        "hyper_title": "Hyperparameters and training parameters",
        "hyper_rows": [[str(cell) for cell in row] for row in HYPER_ROWS],
        "evidence_title": "Implementation evidence — module-wise results",
        "evidence_images": [str(ROOT / "docs" / "assets" / name) for name in
                            ("mod1_data.png", "mod2_train.png", "mod3_latent.png", "mod4_tests.png")],
        "evidence_bullets": ["• " + line for line in EVIDENCE_BULLETS],
        "dashboard": str(ROOT / "docs" / "assets" / "dashboard.png"),
        "repo": "https://github.com/dakshadev/7th-sem",
        "demo_bullets": [
            "Dashboard rebuilt from results/person1/*.json: six conditions, two budgets, geometry, ambiguity.",
            "Walkthrough: build batches, train 40 updates, probe, geometry - all from committed scripts.",
            "Repository: github.com/dakshadev/7th-sem (link below and on the contents slide).",
            "Demo recording: to be captured before the review and stored in docs/assets - not claimed as done here.",
        ],
        "outline": [
            "Introduction — why latent targets, and why small budgets change the answer",
            "Problem statement — the call, and the research question",
            "Literature survey — 15 papers in three tables, and the approved gaps G1–G5",
            "Approved objectives, and progress against each one",
            "Methodology — five modules, and the hyperparameters they run on",
            "Dataset description and computing requirements",
            "Implementation evidence — module-wise results, then preliminary results",
            "Live demonstration — dashboard and repository walkthrough",
            "Remaining work, risks and SDG alignment",
            "Conclusion and references",
        ],
        # Presentation copy: one claim per line, because full sentences force the type down to a
        # size nobody in the room can read. The paper keeps the long-form version.
        "slide_text": {
            "s3": [
                "A text encoder turns language into vectors used for search, ranking and classification.",
                "Two ways to train it without labels: guess the hidden word, or guess a hidden representation.",
                "Word-guessing is noisy: a third of masked positions have five or more valid answers.",
                "Recipes differ in target, masking, predictor and loss at once, so no study isolates the cause.",
                "Our question: which design choice decides success or collapse at small scale?",
            ],
            "s4": [
                "Latent prediction works in vision; in text the evidence conflicts — success in one recipe, collapse in another.",
                "Every published recipe changes several design choices at once, so the cause is untested.",
                "Research question: which choice decides whether a small text encoder collapses or learns transferable representations?",
                "Scope: one encoder, identical data and budget, free GPU only, every claim backed by a stored result.",
            ],
            "s8": [
                "G1 · Confounded comparisons — recipes differ in target, mask, predictor and loss at once.",
                "G2 · Latent-only encoders under-studied — language JEPA targets decoders, not encoders.",
                "G3 · Collapse diagnosed, not remedied — the remedies were never tested on masked text.",
                "G4 · Unreliable evaluation — loss alone hides quality.",
                "G5 · Small-budget regime untested — all text evidence comes from large budgets.",
            ],
            "s14": [
                "No HPC cluster: all GPU work ran on a free Kaggle T4 inside a 30 GPU-hour weekly quota.",
                "Measured: 24 training runs cost 1.2 GPU-hours in total - 35 s at 0.69M tokens, 353 s at 5M.",
                "Minimum viable hardware: one CUDA GPU with 8 GB or more; single-GPU, fully checkpointed.",
                "Lab access would remove the two real constraints: the weekly quota and session resets.",
            ],
            "s18": [
                "Done: 24 runs in two sweeps - the budget sweep (C/A/B) and the anti-collapse sweep (H, I, J).",
                "Next: probe the same checkpoints on masked versus clean input to test the mismatch.",
                "Then: ablations D-G at full budget, and a second probe task so one task cannot carry the result.",
                "Risks: free-tier quota and session resets, and the honest chance that nothing beats an untrained encoder.",
            ],
            "s19": [
                "SDG 4 - Quality Education: lowers the compute cost of usable text representations.",
                "SDG 9 - Industry and Innovation: a tested, reproducible pipeline on free hardware.",
                "SDG 10 - Reduced Inequalities: language technology should not need a large budget.",
                "Secondary - SDG 12: smaller models mean less energy per experiment.",
            ],
            "s20": [
                "A complete, tested system: one encoder, six conditions, 64 tests, 24 GPU runs.",
                "No condition beat an untrained encoder; a 7x larger budget made transfer worse.",
                "Anti-collapse guards and removing the teacher did not close the gap.",
                "A testable negative result - the masked-versus-clean probe decides it.",
            ],
        },
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
