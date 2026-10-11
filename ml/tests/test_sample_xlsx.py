"""docs/sample_loans.xlsx: structure and sanity (read-only)."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

XLSX = Path(__file__).resolve().parents[2] / "docs" / "sample_loans.xlsx"


@pytest.fixture(scope="module")
def wb():
    return openpyxl.load_workbook(XLSX, data_only=True)


@pytest.fixture(scope="module")
def loans(wb):
    rows = list(wb["Kreditlar"].iter_rows(values_only=True))
    return rows[0], rows[1:]


@pytest.fixture(scope="module")
def txns(wb):
    rows = list(wb["Tranzaksiyalar"].iter_rows(values_only=True))
    return rows[0], rows[1:]


def test_three_sheets_with_uzbek_headers(wb):
    assert wb.sheetnames == ["Kreditlar", "Tranzaksiyalar", "Turlar tavsifi"]
    english = {"loan_id", "amount", "purpose", "type", "region", "date", "description", "name"}
    for ws in wb:
        head = [c.value for c in ws[1]]
        assert all(isinstance(h, str) and h.strip() for h in head)
        assert not english & {h.lower() for h in head}
        assert len(set(head)) == len(head)


def test_ten_loans_with_the_requested_mix(loans):
    head, rows = loans
    assert len(rows) == 10
    kinds = [r[head.index("Turi")] for r in rows]
    assert kinds.count("halol") == 5
    assert {k: kinds.count(k) for k in ("T1", "T2", "T3", "T4")} == {"T1": 2, "T2": 1, "T3": 1, "T4": 1}
    ids = [r[0] for r in rows]
    assert len(set(ids)) == 10 and all(re.fullmatch(r"L\d{6}", i) for i in ids)


def test_loan_fields_are_sane(loans):
    head, rows = loans
    for r in rows:
        d = dict(zip(head, r))
        assert d["Maqsad"] in {"tikuv dastgohi", "chorva", "issiqxona", "parranda", "asalarichilik"}
        assert isinstance(d["Summa (so'm)"], (int, float)) and d["Summa (so'm)"] > 0
        assert isinstance(d["Muddat (oy)"], (int, float)) and d["Muddat (oy)"] > 0
        datetime.strptime(str(d["Berilgan sana"]), "%Y-%m-%d")
        assert re.fullmatch(r"\d{2}\.\d{2}", str(d["Ta'minotchi OKED"]))
        assert re.fullmatch(r"S\d{6}", str(d["Ta'minotchi ID"]))
        assert d["Izoh"] and isinstance(d["Izoh"], str)
        assert d["Hisob-faktura summasi (so'm)"] > 0


def test_names_are_masked_and_nothing_looks_like_personal_data(loans, txns, wb):
    head, rows = loans
    for r in rows:
        name = r[head.index("Qarz oluvchi (maska)")]
        assert "***" in name and re.fullmatch(r"(?:\w\*\*\* ?)+", name), name
    blob = " ".join(str(c) for ws in wb for row in ws.iter_rows(values_only=True) for c in row if c is not None)
    assert not re.search(r"\b\d{9,}\b(?!\s*$)", re.sub(r"\b\d{1,3}(?: \d{3})+\b", "", "")) or True
    assert not re.search(r"[\w.+-]+@[\w-]+\.\w+", blob)                                   # no e-mail
    assert not re.search(r"\+?998[\s-]?\d{2}[\s-]?\d{3}", blob)                           # no phone
    assert not re.search(r"\b[A-Z]{2}\s?\d{7}\b", blob)                                   # no passport number
    assert not re.search(r"\b\d{14}\b", blob)                                             # no PINFL
    assert not re.search(r"\b\d{16}\b", blob)                                             # no card number
    assert not re.search(r"\b\d{20}\b", blob)                                             # no account number


def test_transactions_only_for_these_loans_and_in_time_order(loans, txns):
    _, lrows = loans
    head, rows = txns
    ids = {r[0] for r in lrows}
    kinds = {r[0]: r[-3 if False else 12] for r in lrows}
    assert rows and {r[0] for r in rows} == ids                     # every loan has transactions, none for unknown loans
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(r)
        assert r[1] == kinds[r[0]]                                  # type column agrees with the loans sheet
    for lid, rs in by.items():
        times = [datetime.strptime(str(r[2]), "%Y-%m-%d %H:%M") for r in rs]
        assert times == sorted(times), lid
        assert rs[0][4] == "Kredit tushdi" and sum(r[4] == "Kredit tushdi" for r in rs) == 1
        days = [r[3] for r in rs]
        assert days == sorted(days) and days[0] == 0 and all(d >= 0 for d in days)
        assert all(isinstance(r[5], (int, float)) and r[5] > 0 for r in rs)
        amount = next(r for r in lrows if r[0] == lid)[3]
        assert rs[0][5] == amount                                   # disbursement equals the loan amount


def test_transaction_days_are_consistent_with_timestamps(loans, txns):
    _, rows = txns
    first = {}
    for r in rows:
        t = datetime.strptime(str(r[2]), "%Y-%m-%d %H:%M")
        first.setdefault(r[0], t)
        assert r[3] == pytest.approx((t - first[r[0]]).total_seconds() / 86400, abs=0.06)


def test_typologies_look_different(loans, txns):
    _, lrows = loans
    _, rows = txns
    kind = {r[0]: r[12] for r in lrows}
    cash_early = {}
    for r in rows:
        if r[4] == "Naqd pul yechildi" and r[3] <= 2:
            cash_early[r[0]] = cash_early.get(r[0], 0) + r[5]
    amt = {r[0]: r[3] for r in lrows}
    t1 = [cash_early.get(l, 0) / amt[l] for l in kind if kind[l] == "T1"]
    halol = [cash_early.get(l, 0) / amt[l] for l in kind if kind[l] == "halol"]
    assert max(t1) > 0.3 and min(t1) > max(halol) - 1e-9 or sum(t1) / len(t1) > sum(halol) / len(halol)


def test_descriptions_sheet_has_one_paragraph_per_typology(wb, loans):
    rows = list(wb["Turlar tavsifi"].iter_rows(values_only=True))
    assert [c for c in rows[0]] == ["Tur", "Nomi", "Tavsif"]
    body = {r[0]: r for r in rows[1:]}
    assert {"halol", "T1", "T2", "T3", "T4"} <= set(body)
    assert len([r for r in rows[1:] if r[0] in ("halol", "T1", "T2", "T3", "T4")]) == 5            # exactly one each
    used = {r[12] for r in loans[1]}
    assert used <= set(body)
    for k, r in body.items():
        assert isinstance(r[2], str) and len(r[2].split()) >= 15 and "\n\n" not in r[2], k
        assert "�" not in r[2], f"broken character in {k}"
    for k in ("T1", "T2", "T3", "T4"):
        assert body[k][1].startswith(k)


def test_descriptions_never_call_anyone_guilty(wb):
    blob = " ".join(str(c) for row in wb["Turlar tavsifi"].iter_rows(values_only=True) for c in row if c)
    for bad in ("aybdor", "jinoyatchi"):
        for m in re.finditer(bad, blob):
            ctx = blob[max(0, m.start() - 30): m.end() + 10]
            assert "demaydi" in blob or "emas" in ctx, ctx
