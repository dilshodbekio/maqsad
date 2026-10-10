"""Tests of the photo module: ml.api.check_photo / reset_sandbox and the portfolio photo index."""
from __future__ import annotations

import builtins
import io
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml import api
from ml.errors import InvalidImageError
from ml.money_trail.service import get_portfolio
from ml.photo import config, imageio
from ml.photo.faces import FaceDetector
from ml.photo.index import get_index
from ml.photo.transforms import TRANSFORMS

ASSETS = config.ASSETS


def img_bytes(name: str) -> bytes:
    return (ASSETS / name).read_bytes()


@pytest.fixture(scope="module")
def P():
    return get_portfolio()


@pytest.fixture(scope="module")
def idx():
    return get_index()


def loans_without_photo(P, idx, n=3):
    return [l for l in P.loans.index if l not in idx.by_loan][:n]


# ---------------------------------------------------------------- validation
def test_invalid_inputs_raise(P):
    lid = P.loans.index[0]
    jpg = img_bytes("chorva_00.jpg")
    gif = io.BytesIO()
    Image.new("RGB", (40, 40), "red").save(gif, "GIF")
    for bad in (b"", None, "not bytes", b"definitely not an image", jpg[:200], b"\x00" * (config.MAX_BYTES + 1),
                gif.getvalue(), b"%PDF-1.4 fake"):
        with pytest.raises(InvalidImageError):
            api.check_photo(lid, bad)


def test_unknown_loan(P):
    with pytest.raises(api.LoanNotFoundError):
        api.check_photo("L-nope", img_bytes("chorva_00.jpg"))


# ---------------------------------------------------------------- output image
def _exif_jpeg(size=(200, 100)) -> bytes:
    im = Image.new("RGB", size, (30, 120, 200))
    ex = Image.Exif()
    ex[0x0112] = 6                      # orientation: rotate 270 to display
    ex[0x010F] = "SecretCameraMaker"    # Make
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=ex)
    return buf.getvalue()


def test_blurred_image_is_jpeg_without_exif_and_oriented(P):
    src = _exif_jpeg()
    assert b"SecretCameraMaker" in src
    r = api.check_photo(P.loans.index[0], src)
    out = r["blurred_image"]
    assert isinstance(out, bytes) and out[:2] == b"\xff\xd8"
    im = Image.open(io.BytesIO(out))
    assert im.format == "JPEG" and len(im.getexif()) == 0 and "exif" not in im.info
    assert b"SecretCameraMaker" not in out and b"Exif" not in out[:64]
    assert im.size == (100, 200)         # EXIF orientation applied, then pixels only
    assert set(r) == {"asset_match", "duplicate_of", "geo_ok", "time_ok", "faces_blurred", "faces_count",
                      "blurred_image", "latency_ms"}
    assert r["asset_match"] is None and r["latency_ms"] > 0


def test_large_image_is_downscaled(P):
    big = Image.fromarray(np.random.default_rng(0).integers(0, 255, (2400, 3000, 3), dtype=np.uint8))
    buf = io.BytesIO()
    big.save(buf, "JPEG", quality=60)
    out = Image.open(io.BytesIO(api.check_photo(P.loans.index[0], buf.getvalue())["blurred_image"]))
    assert max(out.size) <= config.MAX_SIDE


def test_faces_are_blurred_and_not_redetected(P):
    det = FaceDetector()
    r = api.check_photo(P.loans.index[0], img_bytes("faces_test_09.jpg"))
    assert r["faces_count"] >= 3 and r["faces_blurred"] is True
    blurred = imageio.decode(r["blurred_image"])
    assert det.detect(blurred) == []
    clean = api.check_photo(P.loans.index[0], img_bytes("issiqxona_00.jpg"))
    assert clean["faces_count"] == 0 and clean["faces_blurred"] is False


# ---------------------------------------------------------------- duplicates and sandboxes
def test_portfolio_image_reupload_is_found(P, idx):
    single = next(i for i in idx.images if len(i["loans"]) == 1)
    owner = single["loans"][0]
    other = loans_without_photo(P, idx, 1)[0]
    r = api.check_photo(other, img_bytes(single["file"]))
    assert owner in {d["loan_id"] for d in r["duplicate_of"]}
    assert all(d["similarity"] >= 1 - idx.max_distance for d in r["duplicate_of"])
    # a recompressed + resized copy is found as well
    rng = np.random.default_rng(1)
    rgb = imageio.decode(img_bytes(single["file"]))
    for name in ("JPEG q=30", "resize 50%", "brightness +30%"):
        copy = imageio.encode_jpeg(TRANSFORMS[name](rgb, rng))
        assert owner in {d["loan_id"] for d in api.check_photo(other, copy)["duplicate_of"]}, name
    # the owner's own loan is not reported as a duplicate of itself
    assert owner not in {d["loan_id"] for d in api.check_photo(owner, img_bytes(single["file"]))["duplicate_of"]}


def test_shared_portfolio_images_list_each_other(P, idx):
    shared = [i for i in idx.images if len(i["loans"]) > 1]
    assert len(shared) == 10
    for im in shared:
        photos = {p["photo_id"]: p for l in im["loans"] for p in api.get_loan(l)["photos"]}
        assert im["photo_id"] in photos
        for l in im["loans"]:
            dups = {d["loan_id"] for d in api.get_loan(l)["photos"][0]["duplicate_of"]}
            assert dups == set(im["loans"]) - {l}


def test_unrelated_image_has_no_duplicates(P, idx):
    reserved = idx.reserved[0]
    r = api.check_photo(P.loans.index[0], img_bytes(reserved))
    assert r["duplicate_of"] == []


def test_sandbox_isolation_and_reset(P, idx):
    new = img_bytes(idx.reserved[1])                        # not in the portfolio index
    a, b = "sbx-A", "sbx-B"
    la, lb, lc = loans_without_photo(P, idx, 3)
    api.reset_sandbox(a), api.reset_sandbox(b)
    n_index = len(get_index().images)
    assert api.check_photo(la, new, sandbox_id=a)["duplicate_of"] == []       # first upload in A
    assert [d["loan_id"] for d in api.check_photo(lb, new, sandbox_id=a)["duplicate_of"]] == [la]
    assert api.check_photo(lb, new, sandbox_id=b)["duplicate_of"] == []       # sandbox B does not see A
    assert api.check_photo(lb, new)["duplicate_of"] == []                     # portfolio mode does not see A
    assert api.check_photo(lc, img_bytes(idx.reserved[1]), sandbox_id=None)["duplicate_of"] == []
    # sandbox uploads never enter the portfolio index nor get_loan
    assert len(get_index().images) == n_index
    assert api.get_loan(la)["photos"] == []
    api.reset_sandbox(a)
    assert api.check_photo(lc, new, sandbox_id=a)["duplicate_of"] == []       # reset worked
    api.reset_sandbox("never-used")                                           # idempotent
    api.reset_sandbox(a), api.reset_sandbox(b)


def test_reset_sandbox_none_clears_all_but_not_the_portfolio(P, idx):
    new = img_bytes(idx.reserved[2])
    la, lb, lc = loans_without_photo(P, idx, 3)
    n_index = len(get_index().images)
    for sb in ("sbx-1", "sbx-2"):
        api.check_photo(la, new, sandbox_id=sb)
    assert [d["loan_id"] for d in api.check_photo(lb, new, sandbox_id="sbx-1")["duplicate_of"]] == [la]
    assert [d["loan_id"] for d in api.check_photo(lb, new, sandbox_id="sbx-2")["duplicate_of"]] == [la]
    api.reset_sandbox(None)                                               # clears ALL sandboxes
    for sb in ("sbx-1", "sbx-2"):
        assert api.check_photo(lc, new, sandbox_id=sb)["duplicate_of"] == []
    # portfolio index untouched: portfolio photos are still found
    single = next(i for i in idx.images if len(i["loans"]) == 1)
    assert single["loans"][0] in {d["loan_id"] for d in api.check_photo(lc, img_bytes(single["file"]))["duplicate_of"]}
    assert len(get_index().images) == n_index
    api.reset_sandbox(None)


# ---------------------------------------------------------------- geo / time
def test_geo_and_time_checks(P):
    lid = P.loans.index[0]
    row = P.loans.loc[lid]
    jpg = img_bytes("parranda_00.jpg")
    lat, lon = float(row["borrower_lat"]), float(row["borrower_lon"])
    assert api.check_photo(lid, jpg, lat=lat, lon=lon)["geo_ok"] is True
    assert api.check_photo(lid, jpg, lat=lat + 1.0, lon=lon)["geo_ok"] is False      # ~111 km away
    assert api.check_photo(lid, jpg, lat=lat)["geo_ok"] is None                      # no coordinates
    assert api.check_photo(lid, jpg)["geo_ok"] is None
    assert api.check_photo(lid, jpg, lat=123.0, lon=lon)["geo_ok"] is False          # impossible coordinates
    now = datetime.now(timezone.utc)
    assert api.check_photo(lid, jpg, captured_at=now - timedelta(minutes=1))["time_ok"] is True
    assert api.check_photo(lid, jpg, captured_at=now + timedelta(hours=2))["time_ok"] is False
    assert api.check_photo(lid, jpg, captured_at=now + timedelta(minutes=5))["time_ok"] is True    # within skew
    before = row["disbursed_at"].to_pydatetime() - timedelta(days=1)
    assert api.check_photo(lid, jpg, captured_at=before)["time_ok"] is False
    assert api.check_photo(lid, jpg)["time_ok"] is None


def test_photo_result_does_not_change_risk(P):
    lid = P.loans.index[5]
    before = api.score_loan(lid)
    api.check_photo(lid, img_bytes("chorva_03.jpg"), lat=1.0, lon=1.0, captured_at=datetime.now(timezone.utc))
    after = api.score_loan(lid)
    assert (before["risk"], before["color"], before["reasons"]) == (after["risk"], after["color"], after["reasons"])


# ---------------------------------------------------------------- nothing is written
def _tree_state(*roots: Path) -> dict:
    out = {}
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
                st = p.stat()
                out[str(p)] = (st.st_size, st.st_mtime_ns)
    return out


def test_original_bytes_are_never_written_to_disk(P, monkeypatch):
    import tempfile
    tmp = Path(tempfile.gettempdir())
    repo = config.ROOT
    before = (set(os.listdir(tmp)), _tree_state(repo / "ml", repo / "data"))
    real_open = builtins.open
    writes = []

    def guarded(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            writes.append((file, mode))
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(Path, "write_bytes", lambda *a, **k: writes.append(("write_bytes", a)))
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: writes.append(("write_text", a)))
    api.check_photo(P.loans.index[0], img_bytes("faces_test_09.jpg"), sandbox_id="sbx-disk")
    api.reset_sandbox("sbx-disk")
    monkeypatch.undo()
    assert writes == []
    assert (set(os.listdir(tmp)), _tree_state(repo / "ml", repo / "data")) == before


# ---------------------------------------------------------------- index and get_loan
def test_photo_index_consistency(P, idx):
    raw_files = {p.name for p in ASSETS.glob("*.jpg")}
    loans_with = set(idx.by_loan)
    assert 100 <= len(loans_with) <= 150
    shared = [i for i in idx.images if len(i["loans"]) > 1]
    assert 8 <= len(shared) <= 12 and all(2 <= len(i["loans"]) <= 3 for i in shared)
    assert len({i["file"] for i in idx.images}) == len(idx.images)
    for im in idx.images:
        assert im["file"] in raw_files and set(im["loans"]) <= set(P.loans.index)
        assert all(str(P.loans.at[l, "purpose"]) == im["purpose"] for l in im["loans"])
    assert not set(idx.reserved) & {i["file"] for i in idx.images}


def test_get_loan_photos(P, idx):
    lid = next(l for l in idx.by_loan)
    d = api.get_loan(lid)
    assert len(d["photos"]) == 1
    p = d["photos"][0]
    assert set(p) == {"photo_id", "path", "purpose", "captured_at", "lat", "lon", "duplicate_of", "source"}
    assert p["source"] == "portfolio"
    assert (config.ROOT / p["path"]).is_file() and p["purpose"] == str(P.loans.at[lid, "purpose"])
    for l in loans_without_photo(P, idx, 3):
        assert api.get_loan(l)["photos"] == []
