"""Portfolio photo index: which demo-portfolio loans have a photo, and the perceptual hashes of those photos.

    .venv/Scripts/python -m ml.photo.index        # writes ml/artifacts/photo_index_v1.json

Only a subset of loans has a photo (a borrower uploads one when asked): per purpose 2 images are shared by
2-3 loans (taken from the generator's own photo-reuse groups of the demo portfolio = demo split, seed 7) and
16 images belong to one loan each; 2 images per purpose are kept out of the index (used as "new photo" in tests).
Deterministic (fixed seed). The index stores image -> loans and hashes only; capture time / location of a
loan's photo come from the loan itself at serve time. Sandbox uploads are never added here.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache

import numpy as np

from ml.generator.config import PURPOSES

from . import config, hashing, imageio

SEED = 20260410
N_GROUPS, N_SINGLE = 2, 16          # per purpose; the remaining images of the purpose stay out of the index


def load_dup_config() -> dict:
    return json.loads(config.DUP_PATH.read_text(encoding="utf-8"))


def _hash_file(name: str) -> dict[str, int]:
    return hashing.hash_image(imageio.decode((config.ASSETS / name).read_bytes()))


def build_index() -> dict:
    from ml.generator.build import build
    from ml.generator.config import SPLITS
    t = build(SPLITS["demo"])
    loans = t["loans"].set_index("loan_id")
    purpose = loans["purpose"].astype(str)
    lab = t["labels"].set_index("loan_id")
    by_photo = loans.groupby("photo_id").apply(lambda g: sorted(g.index), include_groups=False)
    groups = {pid: m for pid, m in by_photo.items() if len(m) >= 2 and len({purpose[x] for x in m}) == 1}
    singles = [m[0] for m in by_photo if len(m) == 1]
    in_group = {x for m in groups.values() for x in m}
    rng = np.random.default_rng(SEED)
    files = {p: sorted(r["file"] for r in csv.DictReader(open(config.ASSETS / "manifest.csv", encoding="utf-8"))
                       if r["purpose"] == p) for p in PURPOSES}
    images, reserved = [], []
    for p in PURPOSES:
        fs = list(rng.permutation(files[p]))
        gids = sorted(pid for pid, m in groups.items() if purpose[m[0]] == p)
        gids = [gids[i] for i in rng.permutation(len(gids))[:N_GROUPS]]
        cand = sorted(x for x in singles if purpose[x] == p and x not in in_group)
        picked = [cand[i] for i in rng.permutation(len(cand))[:N_SINGLE]]
        for pid in gids:
            images.append(dict(file=fs.pop(0), purpose=p, photo_id=pid, loans=groups[pid][:3]))
        for lid in picked:
            images.append(dict(file=fs.pop(0), purpose=p, photo_id=loans.at[lid, "photo_id"], loans=[lid]))
        reserved += fs
    for im in images:
        im["hash"] = hashing.to_json(_hash_file(im["file"]))
    n_loans = sorted({x for im in images for x in im["loans"]})
    out = dict(version="photo_index_v1", portfolio=dict(split="demo", seed=SPLITS["demo"].seed),
               seed=SEED, images=images, reserved=sorted(reserved),
               stats=dict(images=len(images), loans=len(n_loans), shared_images=sum(len(i["loans"]) > 1 for i in images),
                          reuse_loans_in_groups=int(sum(bool(lab.at[x, "photo_reused"]) for im in images
                                                        if len(im["loans"]) > 1 for x in im["loans"]))))
    config.INDEX_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    return out


class PhotoIndex:
    """Loaded index + duplicate lookup. Hashes are compared with the variant / threshold of photo_dup_v1.json."""

    def __init__(self):
        raw = json.loads(config.INDEX_PATH.read_text(encoding="utf-8"))
        dup = load_dup_config()
        self.variant, self.max_distance = dup["variant"], dup["max_distance"]
        self.images = [dict(im, hash=hashing.from_json(im["hash"])) for im in raw["images"]]
        self.reserved = raw["reserved"]
        self.by_loan: dict[str, list[dict]] = {}
        for im in self.images:
            for lid in im["loans"]:
                self.by_loan.setdefault(lid, []).append(im)

    def matches(self, h: dict[str, int], exclude_loan: str | None = None) -> list[dict]:
        """Portfolio loans whose photo is a duplicate of hashes `h`: [{loan_id, similarity}], best first."""
        out = []
        for im in self.images:
            d = hashing.distance(self.variant, h, im["hash"])
            if d <= self.max_distance:
                out += [dict(loan_id=lid, similarity=round(1 - d, 4)) for lid in im["loans"] if lid != exclude_loan]
        return sorted(out, key=lambda x: (-x["similarity"], x["loan_id"]))

    def photos_of(self, loan_id: str) -> list[dict]:
        out = []
        for im in self.by_loan.get(loan_id, []):
            out.append(dict(photo_id=im["photo_id"], path=f"ml/assets/photos/{im['file']}", purpose=im["purpose"],
                            duplicate_of=self.matches(im["hash"], exclude_loan=loan_id)))
        return out


@lru_cache(maxsize=1)
def get_index() -> PhotoIndex:
    return PhotoIndex()


if __name__ == "__main__":
    s = build_index()
    print(s["stats"], "reserved", len(s["reserved"]))
