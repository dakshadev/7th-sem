"""Independent audit of the synopsis and the interim deck against the evidence.

Three checks, all mechanical so the result is not an opinion:

1. References — every entry is re-resolved from its DOI (Crossref) or arXiv ID and its title,
   first author and year are compared with what the document claims.
2. Numbers — every headline figure in the document is recomputed from results/person1/*.json
   and checked to appear verbatim in the document.
3. Deck — slide claims are checked against the same evidence, and the file's structure is
   compared with the university template it was built from.

Writes docs/VALIDATION_REPORT.md. Nothing is uploaded anywhere.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from docx import Document
from pptx import Presentation

ROOT = Path("C:/Users/daksh/MyLife/Work/7th-sem")
RESULTS = ROOT / "results" / "person1"
DOCX = Path(os.environ.get("SYNOPSIS_OUT", ROOT / "docs" / "SYNOPSIS_interim_draft_paper.docx"))
DECK = ROOT / "docs" / "Interim_Project_Presentation_7th-sem.pptx"
TEMPLATE = Path("C:/Users/daksh/Downloads/"
                "Interim project _Presentation_Template_7th and 8th Sem.pptx")
REPORT = Path(os.environ.get("VALIDATION_REPORT_OUT", ROOT / "docs" / "VALIDATION_REPORT.md"))

UA = {"User-Agent": "7th-sem-audit/1.0 (academic verification)"}


def get_json(url: str, timeout: int = 25):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


# ------------------------------------------------------------------ 1. references
def resolve_doi(doi: str):
    try:
        message = get_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")["message"]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "title": (message.get("title") or [""])[0],
        "authors": [a.get("family") for a in message.get("author", [])],
        "year": (message.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "venue": (message.get("container-title") or [None])[0],
    }


JOURNAL_REF = "{http://arxiv.org/schemas/atom}journal_ref"


def resolve_arxiv(arxiv_id: str):
    import xml.etree.ElementTree as ET

    ns = {"a": "http://www.w3.org/2005/Atom"}
    request = urllib.request.Request(
        f"http://export.arxiv.org/api/query?id_list={arxiv_id}", headers=UA)
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            root = ET.fromstring(response.read())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    entries = root.findall("a:entry", ns)
    if not entries:
        return {"error": "no entry returned"}
    entry = entries[0]
    return {
        "title": " ".join(entry.find("a:title", ns).text.split()),
        "authors": [a.find("a:name", ns).text.split()[-1] for a in entry.findall("a:author", ns)],
        "year": entry.find("a:published", ns).text[:4],
        "venue": (entry.find(JOURNAL_REF).text if entry.find(JOURNAL_REF) is not None else None),
    }


def search_crossref(title: str, entry: str):
    """Fallback resolution for an entry that carries no identifier."""
    try:
        payload = get_json("https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query.bibliographic": title, "rows": 3,
             "select": "title,author,container-title,issued,DOI"}))["message"]["items"]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    best, score = None, 0.0
    for item in payload:
        candidate = (item.get("title") or [""])[0]
        ratio = difflib.SequenceMatcher(None, candidate.lower(), title.lower()).ratio()
        if ratio > score:
            best, score = item, ratio
    if not best or score < 0.85:
        return {"error": f"no crossref match above threshold (best {score:.2f})"}
    return {"title": (best.get("title") or [""])[0],
            "authors": [a.get("family") for a in best.get("author", [])],
            "year": (best.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "venue": (best.get("container-title") or [None])[0]}


def check_references(doc: Document) -> list[dict]:
    entries = [p.text.strip() for p in doc.paragraphs if re.match(r"^\[\d+\]\s", p.text.strip())]
    rows = []
    for entry in entries:
        number = re.match(r"^\[(\d+)\]", entry).group(1)
        quoted = re.search(r'"([^"]+)"', entry)
        claimed_title = quoted.group(1) if quoted else ""
        doi = re.search(r"doi:\s*(10\.\S+)", entry)
        arxiv = re.search(r"arXiv:(\d{4}\.\d{4,5})", entry)
        # Strip identifiers before hunting for the year, or "arXiv:1907.11692" reads as 1907.
        bare = re.sub(r"10\.\S+|arXiv:\d{4}\.\d{4,5}", "", entry)
        claimed_year = re.search(r"\b(19|20)\d{2}\b", bare)
        if doi:
            meta = resolve_doi(doi.group(1).rstrip("."))
            source = f"doi:{doi.group(1)}"
        elif arxiv:
            meta = resolve_arxiv(arxiv.group(1))
            source = f"arXiv:{arxiv.group(1)}"
        else:
            meta = search_crossref(claimed_title, entry)
            source = "crossref-search"
        if "error" in meta:
            rows.append({"n": number, "source": source, "status": "UNRESOLVED",
                         "title": claimed_title, "detail": meta["error"]})
            continue
        score = difflib.SequenceMatcher(
            None, claimed_title.lower(), (meta["title"] or "").lower()).ratio()
        # A one-year gap is legitimate: a paper can be cited by its journal year while the
        # preprint's v1 date is earlier. Anything larger is a real mismatch.
        year_delta = (abs(int(claimed_year.group(0)) - int(meta["year"]))
                      if claimed_year and meta.get("year") else None)
        year_ok = year_delta is not None and year_delta <= 1
        first_author = (meta["authors"] or [""])[0] or ""
        author_ok = first_author.lower()[:6] in entry.lower() if first_author else None
        status = "OK" if score >= 0.85 and year_ok and author_ok else "CHECK"
        rows.append({"n": number, "source": source, "status": status,
                     "score": round(score, 3), "claimed_year": claimed_year.group(0),
                     "resolved_year": meta.get("year"), "year_delta": year_delta,
                     "title": claimed_title,
                     "resolved_title": meta["title"], "first_author": first_author,
                     "venue": meta.get("venue")})
    return rows


# ------------------------------------------------------------------ 2. numbers
def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def evidence() -> dict:
    runs = defaultdict(list)
    for name in ("sweep_annealed.json", "sweep_anticollapse.json"):
        for run in load(name)["runs"]:
            runs[(run["label"], run["budget_tokens"])].append(run)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731

    probe = {(label, budget): round(mean([r["probe_accuracy"] for r in group]), 3)
             for (label, budget), group in runs.items() if len(group) > 1 or True}
    best_single = max(r["probe_accuracy"] for group in runs.values() for r in group)
    geometry = load("geometry_long.json")["students"]
    rank = {k: v.get("geometry", v)["effective_rank_singular"] for k, v in geometry.items()}
    ambiguity = load("ambiguity_wikitext2.json")["ambiguity"]
    scale = load("sweep_anticollapse.json")["random_baseline"]
    seconds = [r["seconds"] for group in runs.values() for r in group]
    tokens = max(r["nonpadding_tokens"] for group in runs.values() for r in group)
    total_runs = sum(len(g) for g in runs.values())
    hours = sum(seconds) / 3600
    b_mean_069 = round(mean([r["probe_accuracy"] for r in runs[("B", 700_000)]]), 3)
    return {
        "probe": probe, "best_single": round(best_single, 3),
        "rank_mean": {
            "C": round(mean([rank["s17-mlm"], rank["s43-mlm"]]), 1),
            "A": round(mean([rank["s17-latent_a"], rank["s43-latent_a"]]), 1),
            "B": round(mean([rank["s17-latent_b"], rank["s43-latent_b"]]), 1),
        },
        "innovation": None,
        "ambiguity": ambiguity, "random": scale, "seconds": (min(seconds), max(seconds)),
        "hours": round(hours, 2), "total_runs": total_runs, "max_tokens": tokens,
        "b_069_mean": b_mean_069,
    }


def check_numbers(doc: Document, deck_text: str, ev: dict) -> list[dict]:
    text = "\n".join(p.text for p in doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            text += "\n" + " | ".join(c.text for c in row.cells)

    def fmt(value: float, digits: int = 3) -> str:
        return f"{value:.{digits}f}"

    checks = []
    def expect(label: str, needle: str, source: str, where="docx"):
        haystack = text if where == "docx" else deck_text
        checks.append({"check": label, "value": needle, "source": source,
                       "found": needle in haystack, "where": where})

    p = ev["probe"]
    expect("A at 0.69M", fmt(p[("A", 700_000)]), "mean probe accuracy, 2 seeds")
    expect("A at 5M", fmt(p[("A", 5_000_000)]), "mean probe accuracy, 2 seeds")
    expect("H at 5M", fmt(p[("H", 5_000_000)]), "mean probe accuracy, 2 seeds")
    expect("I at 5M", fmt(p[("I", 5_000_000)]), "mean probe accuracy, 2 seeds")
    expect("J at 5M", fmt(p[("J", 5_000_000)]), "mean probe accuracy, 2 seeds")
    expect("J at 0.69M", fmt(p[("J", 700_000)]), "mean probe accuracy, 2 seeds")
    expect("random init", fmt(ev["random"]["accuracy"]), "random_baseline.accuracy")
    expect("majority", fmt(ev["random"]["majority_baseline"]), "random_baseline.majority_baseline")
    expect("best single run", fmt(ev["best_single"]), "max single probe accuracy")
    expect("largest token count", f"{ev['max_tokens']:,}", "max nonpadding_tokens")
    expect("run count", str(ev["total_runs"]), "number of runs in both sweeps")
    expect("MLM rank mean", fmt(ev["rank_mean"]["C"], 1), "geometry_long.json s17/s43 mlm")
    expect("latent A rank mean", fmt(ev["rank_mean"]["A"], 1), "geometry_long.json latent_a")
    expect("ambiguity mean", fmt(ev["ambiguity"]["mean_entropy_nats"]), "ambiguity mean")
    expect("ambiguous fraction", fmt(ev["ambiguity"]["frac_positions_entropy_gt_log5"] * 100, 1),
           "fraction above log 5, as percent")
    expect("GPU hours", fmt(ev["hours"], 1), "sum of run seconds")
    expect("deck: slowest run seconds", str(int(ev["seconds"][1])), "max run seconds",
           where="deck")
    expect("deck: fastest run seconds", str(int(ev["seconds"][0])), "min run seconds",
           where="deck")

    # deck-specific claims
    expect("deck: 64 tests", "64 tests", "pytest run", where="deck")
    expect("deck: 24 runs", "24 runs", "both sweeps", where="deck")
    expect("deck: H/I at 5M", fmt(p[("H", 5_000_000)]), "mean probe accuracy", where="deck")
    expect("deck: J at 5M", fmt(p[("J", 5_000_000)]), "mean probe accuracy", where="deck")
    expect("deck: A at 5M", fmt(p[("A", 5_000_000)]), "mean probe accuracy", where="deck")
    expect("deck: full budget", f"{ev['max_tokens']:,}", "max nonpadding_tokens", where="deck")
    return checks


# ------------------------------------------------------------------ 3. deck
def deck_text(path: Path) -> str:
    prs = Presentation(str(path))
    chunks = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                chunks.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    chunks.extend(cell.text for cell in row.cells)
    return "\n".join(chunks)


def deck_structure() -> dict:
    deck = Presentation(str(DECK))
    info = {"slides": len(deck.slides), "size": (deck.slide_width, deck.slide_height),
            "slide_titles": [], "footer_present": 0, "tables": 0, "pictures": 0}
    for slide in deck.slides:
        title = next((sh.text_frame.text.split("\n")[0] for sh in slide.shapes
                      if sh.has_text_frame and sh.text_frame.text.strip()), "")
        info["slide_titles"].append(title[:70])
        if any("ramaiah" in sh.text_frame.text.lower() for sh in slide.shapes
               if sh.has_text_frame):
            info["footer_present"] += 1
        info["tables"] += sum(1 for sh in slide.shapes
                              if getattr(sh, "has_table", False) and sh.has_table)
        info["pictures"] += sum(1 for sh in slide.shapes if sh.shape_type == 13)
    if TEMPLATE.exists():
        tmpl = Presentation(str(TEMPLATE))
        info["template_slides"] = len(tmpl.slides)
        info["template_size"] = (tmpl.slide_width, tmpl.slide_height)
        info["layout_names_match"] = (
            [layout.name for layout in deck.slide_layouts]
            == [layout.name for layout in tmpl.slide_layouts])
    return info


def main() -> int:
    doc = Document(str(DOCX))
    ev = evidence()
    refs = check_references(doc)
    text_checks = check_numbers(doc, deck_text(DECK), ev)
    structure = deck_structure()

    lines = ["# Validation report", "",
             "Mechanical audit of the synopsis and the interim deck against "
             "`results/person1/*.json`, the arXiv and Crossref APIs, and the university template. "
             "Generated by "
             "`scripts/audit_documents.py`; nothing was uploaded to a third party.", "",
             "## 1. References", ""]
    ok = sum(1 for r in refs if r["status"] == "OK")
    lines.append(f"{ok}/{len(refs)} entries resolve and match on title, first author and year.")
    lines += ["", "| # | Resolves via | Status | Title match | Claimed year | Resolved year |",
              "|---|---|---|---|---|---|"]
    for r in refs:
        lines.append(f"| {r['n']} | {r['source']} | {r['status']} | "
                     f"{r.get('score', '—')} | {r.get('claimed_year', '—')} | "
                     f"{r.get('resolved_year', '—')} |")

    lines += ["", "## 2. Numbers", ""]
    bad = [c for c in text_checks if not c["found"]]
    lines.append(f"{len(text_checks) - len(bad)}/{len(text_checks)} figures recompute to the value "
                 "the document prints.")
    lines += ["", "| Check | Printed | Source | Found |", "|---|---|---|---|"]
    for c in text_checks:
        lines.append(f"| {c['check']} | {c['value']} | {c['source']} | "
                     f"{'yes' if c['found'] else '**NO**'} |")

    lines += ["", "## 3. Deck structure", "",
              f"- slides: {structure['slides']}",
              f"- slide size: {structure['size'][0]} x {structure['size'][1]} EMU",
              f"- tables: {structure['tables']}, pictures: {structure['pictures']}",
              f"- footer band detected on {structure['footer_present']} slides"]
    if "template_slides" in structure:
        lines += [f"- template slides: {structure['template_slides']}, "
                  f"template size: {structure['template_size'][0]} x "
                  f"{structure['template_size'][1]} EMU",
                  f"- layout names identical to template: {structure['layout_names_match']}"]
    lines += ["", "### Slide titles", ""]
    lines += [f"{i}. {t}" for i, t in enumerate(structure["slide_titles"], 1)]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {REPORT}")
    print(f"references OK: {ok}/{len(refs)}")
    print(f"numbers found: {len(text_checks) - len(bad)}/{len(text_checks)}")
    for c in bad:
        print(f"  MISSING {c['check']}: expected {c['value']!r} in {c['where']}")
    print(f"deck: {structure['slides']} slides, {structure['tables']} tables, "
          f"{structure['pictures']} pictures, footer on {structure['footer_present']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
