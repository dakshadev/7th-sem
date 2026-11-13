"""Build a single-file results dashboard from the stored evidence, then screenshot it.

No server and no external assets: the page is self-contained so it can sit in the repository and be
opened by anyone. The deck embeds a screenshot of it.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "person1"
OUT = ROOT / "docs" / "dashboard" / "index.html"

CONDITIONS = {
    "C": ("MLM control", "guess the hidden word"),
    "A": ("Latent + EMA", "final layer target, attention predictor"),
    "B": ("Latent, top-2", "normalised top-2 layers, linear predictor"),
    "H": ("A + VICReg", "variance and covariance auxiliary"),
    "I": ("A + Barlow", "redundancy-reduction auxiliary"),
    "J": ("A, no teacher", "targets from the student itself"),
}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def collect() -> dict:
    runs = []
    for name in ("sweep_annealed.json", "sweep_anticollapse.json"):
        data = load(name)
        for run in data["runs"]:
            runs.append(run)
    probe_mean = {}
    for run in runs:
        probe_mean.setdefault((run["label"], run["budget_tokens"]), []).append(run["probe_accuracy"])
    for key, values in probe_mean.items():
        probe_mean[key] = sum(values) / len(values)
    return {
        "runs": runs,
        "probe": probe_mean,
        "random": load("sweep_anticollapse.json")["random_baseline"]["accuracy"],
        "ambiguity": load("ambiguity_wikitext2.json")["ambiguity"],
        "geometry": load("geometry_long.json")["students"],
    }


def page(data: dict) -> str:
    runs = data["runs"]
    total_tokens = max(r["nonpadding_tokens"] for r in runs)
    best = max(data["probe"].items(), key=lambda item: item[1])
    chart = base64.b64encode((ROOT / "docs" / "figures" / "fig_ablate.png").read_bytes()).decode()

    rows = []
    for label in ("C", "A", "B", "H", "I", "J"):
        low = data["probe"].get((label, 700_000))
        high = data["probe"].get((label, 5_000_000))
        name, description = CONDITIONS[label]
        rows.append(
            f"<tr><td><b>{label}</b></td><td>{name}<div class='sub'>{description}</div></td>"
            f"<td class='num'>{low:.3f}</td><td class='num'>{high:.3f}</td>"
            f"<td class='num {'bad' if high < data['random'] else 'good'}'>{high - data['random']:+.3f}</td></tr>")

    geom = []
    for name, record in sorted(data["geometry"].items()):
        detail = record.get("geometry", record)
        geom.append(f"<tr><td>{name}</td><td class='num'>{detail['effective_rank_singular']:.1f}</td>"
                    f"<td class='num'>{detail['effective_rank_covariance']:.2f}</td>"
                    f"<td>{detail.get('status', '')}</td></tr>")

    ambiguity = data["ambiguity"]
    bands = "".join(
        f"<tr><td>{band}</td><td class='num'>{record['positions']:,}</td>"
        f"<td class='num'>{record['mean_entropy']:.3f}</td></tr>"
        for band, record in ambiguity["entropy_by_frequency"].items())

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>7th-sem results dashboard</title>
<style>
 body {{ margin:0; background:#0f1419; color:#e6edf3; font-family:"Segoe UI",Arial,sans-serif; }}
 .wrap {{ padding:26px 34px; max-width:1480px; }}
 header {{ display:flex; justify-content:space-between; align-items:flex-end;
          border-bottom:1px solid #26313c; padding-bottom:14px; margin-bottom:20px; }}
 h1 {{ font-size:23px; margin:0; letter-spacing:-.3px; }}
 .sub {{ color:#8b98a5; font-size:11px; }}
 .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:22px; }}
 .kpi {{ background:#161d24; border:1px solid #26313c; border-radius:5px; padding:12px 14px; }}
 .kpi .v {{ font-size:26px; font-weight:700; letter-spacing:-.5px; }}
 .kpi .l {{ font-size:10.5px; color:#8b98a5; margin-top:3px; text-transform:uppercase; letter-spacing:.6px; }}
 .cols {{ display:grid; grid-template-columns:1.35fr 1fr; gap:18px; }}
 .card {{ background:#161d24; border:1px solid #26313c; border-radius:5px; padding:14px 16px; margin-bottom:16px; }}
 .card h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.8px; color:#8b98a5; margin:0 0 10px; }}
 table {{ width:100%; border-collapse:collapse; font-size:13px; }}
 th {{ text-align:left; color:#8b98a5; font-weight:600; font-size:10.5px; text-transform:uppercase;
       letter-spacing:.5px; border-bottom:1px solid #26313c; padding:5px 6px; }}
 td {{ padding:6px; border-bottom:1px solid #1d252d; vertical-align:top; }}
 .num {{ text-align:right; font-variant-numeric:tabular-nums; font-family:Consolas,monospace; }}
 .bad {{ color:#f0883e; }} .good {{ color:#3fb950; }}
 img {{ width:100%; max-height:340px; object-fit:contain; border-radius:4px; background:#fff; }}
 .foot {{ color:#6e7b8a; font-size:10px; margin-top:8px; }}
</style></head><body><div class="wrap">
<header>
  <div><h1>Masked latent prediction — results dashboard</h1>
  <div class="sub">WikiText-103 · 4-layer encoder · free Kaggle T4 · github.com/dakshadev/7th-sem</div></div>
  <div class="sub">24 runs · 2 budgets · 2 seeds · commit 2a54edc</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">24</div><div class="l">training runs</div></div>
  <div class="kpi"><div class="v">{total_tokens:,}</div><div class="l">largest token budget</div></div>
  <div class="kpi"><div class="v">{best[1]:.3f}</div><div class="l">best probe accuracy ({best[0][0]})</div></div>
  <div class="kpi"><div class="v">{data['random']:.3f}</div><div class="l">untrained encoder</div></div>
</div>

<div class="cols">
  <div>
    <div class="card"><h2>Transfer — frozen linear probe, AG News</h2>
      <table><tr><th>ID</th><th>Condition</th><th class="num">0.69M</th><th class="num">5M</th>
      <th class="num">Δ vs untrained</th></tr>{''.join(rows)}</table>
      <div class="foot">Two seeds per cell. No condition reaches the untrained encoder.</div>
    </div>
    <div class="card"><h2>Budget sweep</h2><img src="data:image/png;base64,{chart}" alt="accuracy by condition"></div>
  </div>
  <div>
    <div class="card"><h2>Representation geometry — effective rank</h2>
      <table><tr><th>Student</th><th class="num">singular</th><th class="num">covariance</th>
      <th>status</th></tr>{''.join(geom)}</table>
      <div class="foot">Nothing collapsed. Rank rises without accuracy following.</div>
    </div>
    <div class="card"><h2>Ambiguity of the token target</h2>
      <table><tr><th>Frequency band</th><th class="num">positions</th><th class="num">mean nats</th></tr>{bands}</table>
      <div class="foot">Mean {ambiguity['mean_entropy_nats']:.3f} nats ·
      {ambiguity['frac_positions_entropy_gt_log5']*100:.1f}% of blanks have five or more plausible answers.</div>
    </div>
    <div class="card"><h2>Provenance</h2>
      <table><tr><td>Corpus</td><td>wikitext-103-raw-v1, 90,000 records</td></tr>
      <tr><td>Encoder</td><td>4 layers, width 256, 4 heads, FFN 1024</td></tr>
      <tr><td>Schedule</td><td>AdamW 3e-4, linear decay to zero</td></tr>
      <tr><td>Evidence</td><td>results/person1/*.json</td></tr></table>
    </div>
  </div>
</div>
</div></body></html>
"""


def main() -> int:
    data = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(data), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
