"""Fetch openly licensed images from Wikimedia Commons (candidates -> curated set).

    .venv/Scripts/python -m ml.photo.fetch_assets candidates  --work <dir>   # search + download thumbnails
    .venv/Scripts/python -m ml.photo.fetch_assets build --work <dir> --keep keep.json

Only CC0 / CC-BY / CC-BY-SA / public-domain files are accepted (no NC, no ND). Every kept file is re-saved as a
JPEG of at most 512 px (EXIF dropped) into ml/assets/photos/ with manifest.csv
(file, purpose, title, author, license, source_url). Candidates are curated by eye (relevance, faces) before `build`.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

API = "https://commons.wikimedia.org/w/api.php"
UA = "MaqsadHackathonBot/0.1 (WIUT hackathon 2026 prototype; https://github.com/dilshodbekio/maqsad) python-urllib"
OUT = Path(__file__).resolve().parents[1] / "assets" / "photos"
MAX_SIDE, QUALITY = 512, 72

QUERIES = {
    "tikuv_dastgohi": ["sewing machine", "industrial sewing machine", "tailor sewing machine", "garment workshop sewing machines"],
    "chorva": ["cow", "cattle herd", "dairy cattle", "cows pasture"],
    "issiqxona": ["greenhouse vegetables", "greenhouse tomatoes", "polytunnel greenhouse", "greenhouse interior"],
    "parranda": ["poultry farm", "chickens farm", "hens", "broiler chickens"],
    "asalarichilik": ["beehives", "apiary", "beekeeping hives", "bee hives field"],
    "faces_test": ["farmer with cow", "woman sewing machine", "beekeeper with hive", "farmer with chickens",
                   "worker greenhouse", "tailor at sewing machine", "man with cattle", "woman feeding chickens"],
}
OK_LICENSE = re.compile(r"^(CC0|Public domain|PD\b|CC[ -]BY(?:[ -]SA)? \d)", re.I)
BAD_LICENSE = re.compile(r"\b(NC|ND)\b", re.I)


def _get(url: str, retries=4) -> bytes:
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=40) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(3 * (k + 1))


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def search(term: str, limit: int = 50) -> list[dict]:
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{term} filetype:bitmap",
        "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": MAX_SIDE})
    pages = json.loads(_get(f"{API}?{q}")).get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [None])[0]
        if not ii or ii["mime"] not in ("image/jpeg", "image/png") or min(ii["width"], ii["height"]) < 400:
            continue
        md = ii.get("extmetadata", {})
        lic = _strip(md.get("LicenseShortName", {}).get("value", ""))
        if not OK_LICENSE.match(lic) or BAD_LICENSE.search(lic):
            continue
        out.append(dict(title=_strip(p["title"]).removeprefix("File:"), author=_strip(md.get("Artist", {}).get("value", "")) or "unknown",
                        license=lic, source_url=ii["descriptionurl"], thumb=ii.get("thumburl") or ii["url"],
                        desc=_strip(md.get("ImageDescription", {}).get("value", ""))[:160]))
    return out


def candidates(work: Path, per_purpose: int = 40):
    work.mkdir(parents=True, exist_ok=True)
    allc = {}
    for purpose, terms in QUERIES.items():
        seen, items = set(), []
        for term in terms:
            for c in search(term):
                if c["source_url"] not in seen:
                    seen.add(c["source_url"])
                    items.append(c)
            time.sleep(1.0)
        items = items[:per_purpose if purpose != "faces_test" else 60]
        for i, c in enumerate(items):
            c["id"] = f"{purpose}_{i:02d}"
            try:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(_get(c["thumb"])))).convert("RGB")
                img.thumbnail((MAX_SIDE, MAX_SIDE))
                img.save(work / f"{c['id']}.jpg", quality=QUALITY, optimize=True)
                c["ok"] = True
            except Exception as e:                                     # skip broken downloads
                c["ok"] = False
            time.sleep(0.6)
        allc[purpose] = [c for c in items if c["ok"]]
        print(purpose, len(allc[purpose]), "candidates")
    (work / "candidates.json").write_text(json.dumps(allc, indent=1), encoding="utf-8")


def build(work: Path, keep: dict[str, list[str]]):
    allc = json.loads((work / "candidates.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for purpose, ids in keep.items():
        by_id = {c["id"]: c for c in allc[purpose]}
        for n, cid in enumerate(ids):
            c = by_id[cid]
            name = f"{purpose}_{n:02d}.jpg"
            Image.open(work / f"{cid}.jpg").convert("RGB").save(OUT / name, quality=QUALITY, optimize=True)
            rows.append([name, purpose, c["title"], c["author"], c["license"], c["source_url"]])
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "purpose", "title", "author", "license", "source_url"])
        w.writerows(rows)
    print(len(rows), "images,", sum(p.stat().st_size for p in OUT.glob("*.jpg")) // 1024, "KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["candidates", "build"])
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--keep", type=Path)
    a = ap.parse_args()
    candidates(a.work) if a.stage == "candidates" else build(a.work, json.loads(a.keep.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
