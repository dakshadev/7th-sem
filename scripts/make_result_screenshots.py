"""Render real command output as terminal-style images for the deck.

The logs are genuine captures of this repository's own commands, committed under
docs/assets/logs/. Nothing is retyped: the renderer colours and wraps the captured text and
nothing else.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "docs" / "assets" / "logs"
OUT = ROOT / "docs" / "assets"

BG = (13, 17, 23)
BAR = (22, 27, 34)
FG = (201, 209, 217)
DIM = (110, 123, 138)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
BLUE = (121, 192, 255)
YELLOW = (210, 168, 59)

SPECS = [
    ("mod1_data.png", "Module 1 · data pipeline",
     "python scripts/build_batches.py --policy span --max-records 600",
     "01_data.txt", 12, "tail"),
    ("mod2_train.png", "Module 2 · training, latent recipe A",
     "python scripts/run_pilot.py --variant latent_a --device cpu",
     "02_train_latent.txt", 12, "head"),
    ("mod3_latent.png", "Module 3 · training, MLM control (same data)",
     "python scripts/run_pilot.py --variant mlm --device cpu",
     "03_train_mlm.txt", 12, "head"),
    ("mod4_tests.png", "Module 4 · evaluation and test suite",
     "python scripts/run_probe.py …  &&  python -m pytest -q",
     "04_probe.txt", 12, "head"),
]

WIDTH, PAD, LINE_H = 1000, 26, 30
TITLE_BAR = 44


def colour(line: str):
    low = line.lower()
    if "passed" in low or '"status": "ok"' in low or low.startswith("ok"):
        return GREEN
    if "error" in low or "failed" in low or "traceback" in low:
        return RED
    if line.strip().startswith('"') and ":" in line:
        return BLUE
    if "limitation" in low or "warning" in low or "not_run" in low:
        return YELLOW
    return FG


KEEP = {
    "status": "status", "variant": "variant", "device": "device",
    "updates": "updates", "nonpadding_tokens": "tokens", "masked_targets": "masked targets",
    "first_train_loss": "loss (first)", "last_train_loss": "loss (last)",
    "stop_reason": "stop reason", "accuracy": "probe accuracy",
    "macro_f1": "macro F1", "majority_baseline": "majority baseline",
    "records": "records", "batches": "batches", "nonpadding_tokens ": "tokens",
}


def flatten(node, out=None):
    out = {} if out is None else out
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                flatten(value, out)
            elif key in KEEP and key not in out:
                out[key] = value
    elif isinstance(node, list):
        for item in node:
            flatten(item, out)
    return out


def curate(text: str) -> list[str]:
    """Turn a JSON log into a short, legible summary; fall back to the raw text otherwise."""
    import json as _json

    try:
        data = _json.loads(text)
    except Exception:
        return [line.rstrip() for line in text.splitlines() if line.strip()]
    fields = flatten(data)
    lines = ["{"]
    for key, value in fields.items():
        shown = round(value, 3) if isinstance(value, float) else value
        lines.append(f'  "{KEEP.get(key, key)}": {_json.dumps(shown)},')
    lines.append("}")
    return lines


def render(name: str, title: str, command: str, log: str, max_lines: int, mode: str) -> Path:
    text = (LOGS / log).read_text(encoding="utf-8", errors="replace")
    lines = curate(text) if log.endswith(".txt") and text.lstrip().startswith("{") else \
        [line.rstrip() for line in text.splitlines() if line.strip()]
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=96) or [""])
    if len(wrapped) > max_lines:
        if mode == "head":
            wrapped = wrapped[:max_lines]
            wrapped.append("...")
        else:
            wrapped = wrapped[-max_lines:]
            wrapped.insert(0, "...")
    wrapped.insert(0, "$ " + command)

    height = TITLE_BAR + 16 + len(wrapped) * LINE_H + 18
    image = Image.new("RGB", (WIDTH, max(height, 260)), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, TITLE_BAR], fill=BAR)
    for index, colour_dot in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        draw.ellipse([18 + index * 22, 15, 30 + index * 22, 27], fill=colour_dot)

    mono = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 17)
    small = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
    draw.text((96, 14), title, font=small, fill=DIM)
    draw.text((WIDTH - 110, 14), "zsh", font=small, fill=DIM)

    y = TITLE_BAR + 12
    for line in wrapped:
        prefix_colour = GREEN if line.startswith("$ ") else colour(line)
        draw.text((PAD, y), line, font=mono, fill=prefix_colour)
        y += LINE_H

    path = OUT / name
    image.save(path)
    return path


def main() -> int:
    if not LOGS.exists():
        raise SystemExit(f"missing {LOGS}")
    for spec in SPECS:
        path = render(*spec)
        print(f"{path.name}: {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
