"""The fusion policy tables in docs/api_contract.md and docs/metrics.md must match ml/fusion.py POLICY (parsed, not copied)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ml import fusion

DOCS = Path(__file__).resolve().parents[2] / "docs"


def table_rows(path, header_first_cell="Code"):
    lines = (DOCS / path).read_text(encoding="utf-8").splitlines()
    rows, on = [], False
    for l in lines:
        if l.startswith("|") and l.split("|")[1].strip() == header_first_cell:
            on, rows = True, []
            continue
        if on and l.startswith("|"):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if set(cells[0]) <= set("-: "):
                continue
            rows.append(cells)
        elif on:
            if any(r and r[0].startswith("`PHOTO_") for r in rows):
                break
            on = False
    return rows


def code_of(cell):
    return cell.strip("`")


@pytest.mark.parametrize("path", ["api_contract.md", "metrics.md"])
def test_policy_table_lists_exactly_the_policy_codes_and_min_colours(path):
    rows = table_rows(path)
    assert rows, f"no policy table in {path}"
    assert {code_of(r[0]) for r in rows} == set(fusion.POLICY)
    for r in rows:
        assert r[1] == fusion.POLICY[code_of(r[0])]["min_color"], (path, r)
    assert len(rows) == len(fusion.POLICY)


def test_metrics_active_column_matches_policy():
    for r in table_rows("metrics.md"):
        assert (r[2] == "yes") is fusion.POLICY[code_of(r[0])]["active"], r


def test_metrics_trigger_text_equals_policy_when():
    for r in table_rows("metrics.md"):
        assert r[3] == fusion.POLICY[code_of(r[0])]["when"], r


def test_api_contract_does_not_say_asset_mismatch_is_inactive_or_asset_match_null():
    t = (DOCS / "api_contract.md").read_text(encoding="utf-8")
    assert fusion.POLICY["PHOTO_ASSET_MISMATCH"]["active"] is True
    assert "inactive until" not in t and "stays inactive" not in t, "api_contract.md still says PHOTO_ASSET_MISMATCH is inactive"
    assert "is `null` until the CLIP step" not in t, "api_contract.md still says asset_match is null until CLIP exists"
    assert "not wired yet" not in t


def test_api_contract_asset_match_shape_lists_all_four_fields():
    t = (DOCS / "api_contract.md").read_text(encoding="utf-8")
    m = re.search(r"asset_match: \{([^}]*)\}", t)
    assert m, "asset_match shape missing"
    fields = {f.strip() for f in m.group(1).split(",")}
    assert fields == {"label", "score", "matches_purpose", "label_uz"}, fields


def test_metrics_has_no_stale_inactive_claim():
    t = (DOCS / "metrics.md").read_text(encoding="utf-8")
    assert "inactive until the CLIP" not in t
