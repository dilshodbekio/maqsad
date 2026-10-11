"""Photo module settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "ml" / "assets" / "photos"
FACES = ROOT / "ml" / "assets" / "faces_test"          # images of real people: local only, git-ignored, never committed or served
ARTIFACTS = ROOT / "ml" / "artifacts"
INDEX_PATH = ARTIFACTS / "photo_index_v1.json"
DUP_PATH = ARTIFACTS / "photo_dup_v1.json"          # duplicate-matching variant + threshold (from ml/eval/eval_photo.py)
YUNET_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"

MAX_BYTES = 10 * 1024 * 1024        # larger uploads are rejected
MAX_SIDE = 1600                     # decoded images are downscaled to this
MAX_PIXELS = 60_000_000             # decompression-bomb guard
JPEG_QUALITY = 85

# ASSUMPTION: a genuine asset photo is taken within this distance of the borrower's registered location.
GEO_MAX_KM = 5.0
FUTURE_SKEW_MIN = 10                # captured_at later than server time + 10 min is not plausible
