"""The ONLY place the backend talks to ML.

`from ml.api import score_loan, check_photo, get_queue, get_loan` works -> ML_MODE = "real".
ImportError -> backend/app/ml_stub.py -> ML_MODE = "stub" (unless REQUIRE_REAL_ML=1: then startup fails).

Optional, not yet approved by ml (docs/api_contract_changes.md #3, #4) — both paths work:
- `sandbox_id` is passed to check_photo only if its signature accepts it.
- `blurred_image` (bytes) is taken from check_photo's result if present; otherwise nothing is stored.
"""
from __future__ import annotations

import inspect
import logging
import os
from types import ModuleType
from typing import Any, Callable, Optional, Tuple

log = logging.getLogger("maqsad.ml")


class MLNotFound(Exception):
    """Unknown loan_id (ml raised KeyError/LookupError)."""


class MLBadInput(Exception):
    """ml rejected the input (ValueError), e.g. an undecodable image."""


class MLFailure(Exception):
    """Any other ml error."""


def _to_builtin(x: Any) -> Any:
    """numpy/pandas scalars and timestamps -> plain JSON-friendly Python."""
    if isinstance(x, dict):
        return {k: _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, (str, bytes, bool, int, float)) or x is None:
        return x
    if hasattr(x, "isoformat"):
        return x.isoformat()
    if hasattr(x, "item"):
        try:
            return x.item()
        except (TypeError, ValueError):
            pass
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _accepts_kw(fn: Callable, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


class MLAdapter:
    def __init__(self, impl: ModuleType, mode: str):
        self.impl = impl
        self.mode = mode
        self._check_photo_takes_sandbox = _accepts_kw(impl.check_photo, "sandbox_id")

    @property
    def model_version(self) -> str:
        return str(getattr(self.impl, "MODEL_VERSION", None) or "unknown")

    def _call(self, name: str, *args, **kwargs):
        try:
            return _to_builtin(getattr(self.impl, name)(*args, **kwargs))
        except LookupError as e:
            raise MLNotFound(str(e)) from e
        except ValueError as e:
            log.warning("ml.%s rejected input: %s", name, e)
            raise MLBadInput(str(e)) from e
        except Exception as e:  # noqa: BLE001 - map everything else to 502
            log.exception("ml.%s failed", name)
            raise MLFailure(f"{type(e).__name__}: {e}") from e

    def get_queue(self) -> list:
        return self._call("get_queue")

    def get_loan(self, loan_id: str) -> dict:
        return self._call("get_loan", loan_id)

    def score_loan(self, loan_id: str) -> dict:
        return self._call("score_loan", loan_id)

    def check_photo(self, loan_id: str, image_bytes: bytes, lat=None, lon=None, captured_at=None,
                    sandbox_id: Optional[str] = None) -> Tuple[dict, Optional[bytes]]:
        """Returns (result without blurred_image, blurred_image bytes or None)."""
        kwargs = dict(lat=lat, lon=lon, captured_at=captured_at)
        if self._check_photo_takes_sandbox:
            kwargs["sandbox_id"] = sandbox_id
        try:
            raw = self.impl.check_photo(loan_id, image_bytes, **kwargs)
        except LookupError as e:
            raise MLNotFound(str(e)) from e
        except ValueError as e:
            log.warning("ml.check_photo rejected input: %s", e)
            raise MLBadInput(str(e)) from e
        except Exception as e:  # noqa: BLE001
            log.exception("ml.check_photo failed")
            raise MLFailure(f"{type(e).__name__}: {e}") from e
        raw = dict(raw)
        blurred = raw.pop("blurred_image", None)
        if blurred is not None and not isinstance(blurred, (bytes, bytearray)):
            log.warning("ml.check_photo blurred_image is %s, not bytes; ignoring", type(blurred).__name__)
            blurred = None
        return _to_builtin(raw), (bytes(blurred) if blurred is not None else None)

    def reset(self) -> None:
        fn = getattr(self.impl, "reset", None)
        if callable(fn):
            fn()


def load_adapter() -> MLAdapter:
    require_real = os.environ.get("REQUIRE_REAL_ML", "0") == "1"
    try:
        if os.environ.get("ML_FORCE_STUB", "0") == "1":
            raise ImportError("ML_FORCE_STUB=1")
        from ml.api import check_photo, get_loan, get_queue, score_loan  # noqa: F401
        import ml.api as impl
        log.info("ML mode: real (ml.api imported)")
        return MLAdapter(impl, "real")
    except ImportError as e:
        if require_real:
            raise RuntimeError(f"REQUIRE_REAL_ML=1 but ml.api could not be imported: {e}") from e
        log.warning("ML mode: STUB (ml.api not importable: %s)", e)
        from . import ml_stub
        return MLAdapter(ml_stub, "stub")
