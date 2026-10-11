"""Asset check: "is the thing in the photo what the loan was for?" - OpenCLIP ViT-B-32 on CPU + a linear probe.

* CLIP image embeddings (512-d, L2-normalised) of the photo; classes = the 5 loan purposes + "other" (everything else:
  rooms, cars, phones, people, food, landscapes, documents ...).
* Zero-shot: cosine similarity to averaged English prompts per class. Linear probe: logistic regression on the embeddings
  of the committed library (ml/artifacts/asset_probe_v1.npz, trained by ml/eval/eval_asset.py; the weights of CLIP itself are
  NOT committed).
* "matches the loan purpose" = probe probability of the loan's purpose >= tau (tau chosen inside cross-validation).

CLIP weights (OpenAI ViT-B-32) live in a cache directory: env `MAQSAD_CLIP_CACHE` (default `ml/.cache/clip`). They are NEVER
downloaded at request time: the service only loads `ViT-B-32.pt` from that directory; a Dockerfile pre-downloads it at build time with
`python -m ml.photo.asset --download` (see ml/README.md). If CLIP cannot be loaded (no torch, no weights, corrupt probe) the failure is cached and logged
once, `check_photo` returns `asset_match: None` without retrying, and no PHOTO_ASSET_MISMATCH is raised. `MAQSAD_ASSET_CHECK=off` switches the check off.
`torch` threads: env `MAQSAD_TORCH_THREADS` (default: torch's default).

Decision rule (one rule for label, matches_purpose and the reason text): `matches_purpose = (argmax class == the loan's purpose) and
p(purpose) >= tau`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from ml.generator.config import PURPOSES

from . import config

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # open_clip must never reach out to the Hugging Face hub at runtime
MODEL_NAME, PRETRAINED = "ViT-B-32", "openai"
WEIGHTS_FILE = "ViT-B-32.pt"
log = logging.getLogger("maqsad.asset")


class AssetUnavailable(RuntimeError):
    """CLIP could not be loaded (cached): check_photo then returns asset_match None."""
CLASSES = (*PURPOSES, "other")
PROBE_NPZ = config.ARTIFACTS / "asset_probe_v1.npz"
PROBE_JSON = config.ARTIFACTS / "asset_probe_v1.json"
DEFAULT_CACHE = config.ROOT / "ml" / ".cache" / "clip"

LABEL_UZ = {"tikuv_dastgohi": "tikuv mashinasi", "chorva": "chorva (qoramol)", "issiqxona": "issiqxona",
            "parranda": "parranda (tovuq)", "asalarichilik": "ari uyasi", "other": "boshqa narsa"}
LABEL_I18N = {
    "tikuv_dastgohi": dict(uz="tikuv mashinasi", en="sewing machine", ru="швейная машина"),
    "chorva": dict(uz="chorva (qoramol)", en="livestock (cattle)", ru="скот (крупный рогатый)"),
    "issiqxona": dict(uz="issiqxona", en="greenhouse", ru="теплица"),
    "parranda": dict(uz="parranda (tovuq)", en="poultry (chickens)", ru="птица (куры)"),
    "asalarichilik": dict(uz="ari uyasi", en="beehive", ru="улей"),
    "other": dict(uz="boshqa narsa", en="something else", ru="другой предмет"),
}
# the Uzbek name of what was found in a non-asset photo -> the same name in uz / en / ru
OTHER_FINE_I18N = {
    "avtomobil": dict(uz="avtomobil", en="car", ru="автомобиль"),
    "transport vositasi": dict(uz="transport vositasi", en="vehicle", ru="транспортное средство"),
    "telefon yoki noutbuk": dict(uz="telefon yoki noutbuk", en="phone or laptop", ru="телефон или ноутбук"),
    "xona": dict(uz="xona", en="room", ru="комната"),
    "mebel": dict(uz="mebel", en="furniture", ru="мебель"),
    "odam": dict(uz="odam", en="person", ru="человек"),
    "oziq-ovqat": dict(uz="oziq-ovqat", en="food", ru="еда"),
    "manzara": dict(uz="manzara", en="landscape", ru="пейзаж"),
    "daraxtlar": dict(uz="daraxtlar", en="trees", ru="деревья"),
    "hujjat": dict(uz="hujjat", en="document", ru="документ"),
    "uy hayvoni": dict(uz="uy hayvoni", en="pet", ru="домашнее животное"),
    "bino": dict(uz="bino", en="building", ru="здание"),
    "velosiped": dict(uz="velosiped", en="bicycle", ru="велосипед"),
}
_BY_UZ = {v["uz"]: v for v in (*LABEL_I18N.values(), *OTHER_FINE_I18N.values())}


def label_i18n(label_uz: str) -> dict:
    """{uz, en, ru} name of what the asset check found, from its Uzbek name (also for results stored in the photo index)."""
    return dict(_BY_UZ.get(label_uz, dict(uz=label_uz, en=label_uz, ru=label_uz)))
PROMPTS = {
    "tikuv_dastgohi": ["a photo of a sewing machine", "a sewing machine on a table", "an industrial sewing machine in a garment workshop"],
    "chorva": ["a photo of a cow", "cattle on a farm", "a herd of cows in a pasture"],
    "issiqxona": ["a photo of a greenhouse", "plants growing inside a greenhouse", "a polytunnel with vegetables"],
    "parranda": ["a photo of chickens", "a poultry farm with hens", "chickens in a chicken coop"],
    "asalarichilik": ["a photo of beehives", "an apiary with bee hives", "a bee hive and bees"],
    "other": ["a photo of an everyday object", "a photo of a room interior", "a photo of a car", "a photo of a person",
              "a photo of a landscape", "a photo of food", "a photo of a document", "a photo of a phone or a laptop"],
}
# only used to NAME what was found when the photo is not an asset ("aniqlangan: avtomobil")
OTHER_FINE = {
    "avtomobil": "a photo of a car", "transport vositasi": "a photo of a bus, a train or an airplane", "telefon yoki noutbuk": "a photo of a phone or a laptop",
    "xona": "a photo of a room interior", "mebel": "a photo of furniture", "odam": "a photo of a person", "oziq-ovqat": "a photo of food",
    "manzara": "a photo of a landscape", "daraxtlar": "a photo of trees in a forest", "hujjat": "a photo of a paper document",
    "uy hayvoni": "a photo of a dog or a cat", "bino": "a photo of a building", "velosiped": "a photo of a bicycle",
}


def cache_dir() -> Path:
    return Path(os.environ.get("MAQSAD_CLIP_CACHE") or DEFAULT_CACHE)


def weights_path() -> Path:
    return cache_dir() / WEIGHTS_FILE


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class AssetModel:
    """CLIP + prompts (+ the probe if its artifact exists). Thread-safe; one instance per process."""

    def __init__(self, allow_download: bool = False):
        if os.environ.get("MAQSAD_ASSET_CHECK", "").lower() in ("off", "0", "false", "no"):
            raise OSError("asset check switched off (MAQSAD_ASSET_CHECK=off)")
        if not allow_download and not weights_path().exists():     # never download at request time
            raise OSError(f"CLIP weights not found at {weights_path()} (pre-download at build time: python -m ml.photo.asset --download)")
        import open_clip
        import torch
        if os.environ.get("MAQSAD_TORCH_THREADS", "").isdigit():
            torch.set_num_threads(int(os.environ["MAQSAD_TORCH_THREADS"]))
        self._torch = torch
        cache_dir().mkdir(parents=True, exist_ok=True)
        # OpenAI weights: open_clip downloads `ViT-B-32.pt` from OpenAI's public URL into the cache dir (sha256 is checked
        # from the URL) or reuses the file that is already there; the Hugging Face route is not used.
        from open_clip.pretrained import get_pretrained_cfg
        self.model = open_clip.openai.load_openai_model(MODEL_NAME, device="cpu", cache_dir=str(cache_dir()))
        cfg = get_pretrained_cfg(MODEL_NAME, PRETRAINED)
        self.preprocess = open_clip.image_transform(self.model.visual.image_size, is_train=False, mean=cfg["mean"], std=cfg["std"],
                                                    interpolation=cfg["interpolation"], resize_mode=cfg["resize_mode"])
        self.model.eval()
        tok = open_clip.get_tokenizer(MODEL_NAME)
        self._lock = threading.Lock()
        self.text = np.stack([self._text(PROMPTS[c], tok) for c in CLASSES])
        self.fine_names = list(OTHER_FINE)
        self.fine_text = np.stack([self._text([OTHER_FINE[n]], tok) for n in self.fine_names])
        self.W = self.b = None
        self.tau = None
        self.meta = {}
        if PROBE_NPZ.exists() and PROBE_JSON.exists():
            try:
                z = np.load(PROBE_NPZ)
                self.W, self.b = z["W"], z["b"]
                self.meta = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
                self.tau = float(self.meta["tau"])
            except (ValueError, KeyError, OSError) as e:
                raise RuntimeError(f"asset probe artifact is corrupt: {type(e).__name__}") from e

    def _text(self, prompts, tok) -> np.ndarray:
        with self._torch.no_grad():
            t = self.model.encode_text(tok(prompts))
            t = t / t.norm(dim=-1, keepdim=True)
            m = t.mean(dim=0)
            return (m / m.norm()).numpy()

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        x = self.preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with self._lock, self._torch.no_grad():
            e = self.model.encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
        return e[0].numpy().astype(np.float64)

    # ---- classifiers on embeddings
    def zero_shot_probs(self, emb: np.ndarray) -> np.ndarray:
        return softmax(100.0 * np.atleast_2d(emb) @ self.text.T)

    def probe_probs(self, emb: np.ndarray) -> np.ndarray:
        return softmax(np.atleast_2d(emb) @ self.W.T + self.b)

    def other_name(self, emb: np.ndarray) -> str:
        return self.fine_names[int(np.argmax(self.fine_text @ emb))]

    def asset_match(self, rgb: np.ndarray, purpose: str) -> dict:
        """{label, score, matches_purpose, label_uz} for a photo uploaded for a loan of this purpose."""
        if self.W is None:
            raise RuntimeError("asset probe artifact missing: run `python -m ml.eval.eval_asset`")
        return self.decide(self.embed(rgb), purpose)

    def decide(self, emb: np.ndarray, purpose: str) -> dict:
        """The decision from an embedding: label = most likely class; matches_purpose = (label == purpose) and p(purpose) >= tau."""
        pi = CLASSES.index(purpose)                       # ValueError for an unknown purpose
        p = self.probe_probs(emb)[0]
        k = int(np.argmax(p))
        label = CLASSES[k]
        label_uz = self.other_name(emb) if label == "other" else LABEL_UZ[label]
        return dict(label=label, score=round(float(p[k]), 4),
                    matches_purpose=bool(k == pi and p[pi] >= self.tau),
                    label_uz=label_uz, label_i18n=label_i18n(label_uz))


_init_lock = threading.Lock()
_model: Optional[AssetModel] = None
_failed: Optional[str] = None


def get_asset_model() -> AssetModel:
    """Load CLIP once (thread-safe) from the local cache. Call it during warm-up (ml.api.warm_up) so no upload pays for the load.
    A failed load is cached and logged ONCE; later calls raise AssetUnavailable immediately (no retry, no download, no blocking)."""
    global _model, _failed
    if _model is not None:
        return _model
    if _failed is not None:
        raise AssetUnavailable(_failed)
    with _init_lock:
        if _model is None and _failed is None:
            try:
                _model = AssetModel()
            except Exception as e:                         # noqa: BLE001 - any failure disables the check, the service keeps running
                _failed = f"{type(e).__name__}: {e}"
                log.warning("asset check disabled for this process (check_photo returns asset_match None): %s", _failed)
    if _model is None:
        raise AssetUnavailable(_failed)
    return _model


def load_error() -> Optional[str]:
    """Why CLIP is not available (None if it is, or was not tried yet)."""
    return _failed


def _reset_for_tests():
    global _model, _failed
    _model, _failed = None, None


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:                     # build time only: the one place that may fetch the weights
        AssetModel(allow_download=True)
        print("CLIP weights cached in", weights_path(), "-", weights_path().stat().st_size // 2 ** 20, "MB")
