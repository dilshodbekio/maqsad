"""Decode / validate / re-encode images entirely in memory (nothing is ever written to disk)."""
from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, ImageOps

from ml.errors import InvalidImageError

from . import config

Image.MAX_IMAGE_PIXELS = config.MAX_PIXELS
ALLOWED = {"JPEG", "PNG", "WEBP", "BMP"}


def decode(data) -> np.ndarray:
    """bytes -> RGB uint8 array, EXIF orientation applied, longest side <= MAX_SIDE.
    Raises InvalidImageError for empty / oversized / not-an-image / corrupted input."""
    if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) == 0:
        raise InvalidImageError("empty image")
    if len(data) > config.MAX_BYTES:
        raise InvalidImageError(f"image larger than {config.MAX_BYTES // (1024 * 1024)} MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(bytes(data)))
            if im.format not in ALLOWED:
                raise InvalidImageError(f"unsupported image format: {im.format}")
            im.load()
            im = ImageOps.exif_transpose(im).convert("RGB")
    except InvalidImageError:
        raise
    except Exception as e:                                    # corrupted, truncated, bomb, not an image
        raise InvalidImageError(f"cannot decode image: {type(e).__name__}") from e
    im.thumbnail((config.MAX_SIDE, config.MAX_SIDE), Image.LANCZOS)
    return np.asarray(im)


def encode_jpeg(rgb: np.ndarray) -> bytes:
    """RGB array -> JPEG bytes. No EXIF / metadata is carried over (pixels only)."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=config.JPEG_QUALITY)
    return buf.getvalue()
