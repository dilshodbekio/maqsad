"""docs/cost.md: structure, sourcing and arithmetic (read-only; nothing is re-measured or rewritten)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "cost.md"
RES = ROOT / "ml" / "artifacts" / "eval_cost.json"
SRC = ROOT / "ml" / "eval" / "measure_cost.py"


@pytest.fixture(scope="module")
def text():
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def res():
    return json.loads(RES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prices():
    from ml.eval.measure_cost import PRICES
    return PRICES


def section(text, n):
    m = re.search(rf"^## {n}\. .*?(?=^## |\Z)", text, re.S | re.M)
    assert m, f"section {n} missing"
    return m.group(0)


def test_sections_present_in_order(text):
    heads = re.findall(r"^## (\d)\.", text, re.M)
    assert heads == ["1", "2", "3", "4", "5", "6"]


def test_every_assumption_line_starts_with_taxmin(text):
    lines = [l for l in section(text, 6).splitlines()[1:] if l.strip()]
    assert len(lines) >= 5
    assert all(l.startswith("- Taxmin: ") for l in lines), [l[:40] for l in lines if not l.startswith("- Taxmin: ")]


def test_taxmin_only_in_assumptions_or_marked_parameters(text):
    outside = text.replace(section(text, 6), "")
    assert "Taxmin" not in outside


def test_both_prices_cite_url_and_date(text, prices):
    s2 = section(text, 2)
    bullets = [l for l in s2.splitlines() if l.startswith("- ")]
    assert len(bullets) == 2
    for b in bullets:
        assert re.search(r"https://\S+", b), b[:60]
        assert prices["read_on"] in b, b[:60]
    assert prices["railway"]["url"] in bullets[0]
    assert prices["dedicated"]["price_source"] in bullets[1] and prices["dedicated"]["spec_url"] in bullets[1]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", prices["read_on"])


def test_unconfirmed_price_is_flagged_honestly(text, prices):
    if not prices["dedicated"]["price_confirmed_on_official_page"]:
        assert "not confirmed" in section(text, 2)


def _numbers(s):
    return re.findall(r"(?<![\w.])(\d+(?:\.\d+)?(?:e-?\d+)?)", s)


def test_every_usd_or_eur_price_is_in_measure_cost_prices_or_derived(text, res, prices):
    src = SRC.read_text(encoding="utf-8")
    R, D = prices["railway"], prices["dedicated"]
    m = res["single_thread"]
    core_s = D["price_eur_per_month"] / 720 / D["cores"] / 3600
    derived = {f"{R['cpu_usd_per_vcpu_minute'] / 60:.3e}", f"{core_s:.3e}", f"{m['rss_mb'] / 1024 * R['ram_usd_per_gb_month']:.2f}"}
    for k in ("score_loan", "score_scenario", "check_photo"):
        c = m[k]["cpu_ms"] / 1000
        derived |= {f"{c * 1000 * R['cpu_usd_per_vcpu_minute'] / 60:.5f}", f"{c * 1000 / 3600 * D['price_eur_per_month'] / 720 / D['cores']:.5f}"}
    found = re.findall(r"\$(\d[\d.,]*(?:e-?\d+)?)|(\d[\d.,]*(?:e-?\d+)?)\s*EUR", text)
    prices_found = [a or b for a, b in found]
    assert len(prices_found) >= 8
    for p in prices_found:
        p = p.rstrip(".,")
        in_src = re.search(rf"(?<![\d.]){re.escape(p)}(?![\d])", src) or re.search(rf"(?<![\d.]){re.escape(p.rstrip('0').rstrip('.') or p)}(?![\d])", src)
        assert in_src or p in derived, f"unsourced price {p!r}"


def test_cpu_times_in_table_match_eval_cost_json(text, res):
    s1 = section(text, 1).split("### 1b")[0]
    m = res["single_thread"]
    expect = [m["score_loan"], m["score_scenario"], m["check_photo"], m["check_photo_parts"]["decode_and_hash"],
              m["check_photo_parts"]["face_detect_and_blur"], m["check_photo_parts"]["clip_asset_check"]]
    rows = [l for l in s1.splitlines() if l.startswith("| ") and "ms |" in l]
    assert len(rows) == len(expect)
    for row, e in zip(rows, expect):
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[1] == f"{e['cpu_ms']:.0f} ms" and cells[2] == f"{e['wall_ms']:.0f} ms", (row, e)
    wu = m["warm_up"]
    assert f"| {wu['cpu_s']:.0f} s | {wu['wall_s']:.0f} s |" in s1
    mc = res["all_cores"]
    s1b = section(text, 1).split("### 1b")[1]
    for k in ("score_loan", "score_scenario", "check_photo"):
        assert f"| {mc[k]['cpu_ms']:.0f} ms | {mc[k]['wall_ms']:.0f} ms |" in s1b


def test_cost_per_1000_reproduces_from_formulas_in_the_doc(text, res):
    s3 = section(text, 3)
    f = re.search(r"Railway = CPU seconds x 1 000 x \$([\d.]+) / 60\. Dedicated = CPU seconds x 1 000 / 3 600 x \(([\d.]+) EUR / (\d+) h / (\d+) cores\)", s3)
    assert f, "formulas missing or reworded"
    per_min, eur, hours, cores = float(f.group(1)), float(f.group(2)), int(f.group(3)), int(f.group(4))
    assert hours == 720
    m = res["single_thread"]
    rows = {l.split("|")[1].strip(): l.split("|") for l in s3.splitlines() if l.startswith("| ") and re.search(r"\d\.\d{5}", l)}
    assert len(rows) == 3
    for key, label in (("score_loan", "Money-trail score"), ("score_scenario", "Scenario score"), ("check_photo", "Photo check (incl. CLIP)")):
        c = m[key]["cpu_ms"] / 1000
        cells = rows[label]
        assert cells[2].strip() == f"{c * 1000 * per_min / 60:.5f}", (label, cells)
        assert cells[3].strip() == f"{c * 1000 / 3600 * (eur / hours / cores):.5f}", (label, cells)
    core_s = eur / hours / cores / 3600
    assert f"{core_s:.3e}" in s3


def test_always_on_memory_cost_reproduces(text, res):
    s4 = section(text, 4)
    gb = res["single_thread"]["rss_mb"] / 1024
    m = re.search(r"([\d.]+) GB x \$([\d.]+) = \*\*\$([\d.]+) per month", s4)
    assert m and float(m.group(1)) == pytest.approx(gb, abs=0.005)
    assert float(m.group(3)) == pytest.approx(gb * float(m.group(2)), abs=0.006)


def test_photo_check_is_more_expensive_than_score_and_parts_are_below_total(res):
    m = res["single_thread"]
    assert m["check_photo"]["cpu_ms"] > m["score_loan"]["cpu_ms"]
    p = m["check_photo_parts"]
    assert sum(v["cpu_ms"] for v in p.values()) <= m["check_photo"]["cpu_ms"] * 1.1
    assert all(v["n"] >= 30 for v in p.values()) and m["check_photo"]["n"] >= 40


def test_llm_cost_is_a_parameter_not_an_invented_number(text):
    s5 = section(text, 5)
    assert "LLM_COST_USD_PER_REPORT" in s5 and "(parameter" in s5
    assert "it is not estimated here" in s5


def test_doc_matches_a_fresh_render_of_the_json(text, res):
    """render() is pure; the committed doc must be exactly what it produces from the committed JSON."""
    from ml.eval.measure_cost import render
    assert render(res["all_cores"], res["single_thread"]) == text
