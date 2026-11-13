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
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
