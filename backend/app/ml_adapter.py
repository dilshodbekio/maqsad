"""The ONLY place the backend talks to ML.

Loading:
- `import ml.api` works -> mode "real". ANY failing import (ImportError; OSError such as a missing
  libgomp for LightGBM; SyntaxError from code needing a newer Python) -> backend/app/ml_stub.py,
  mode "stub" (visible STUB banner).
  With REQUIRE_REAL_ML=1 a failing import stops the service instead.
- Warm-up runs in a background thread (so /api/health answers immediately): the first get_queue()
  builds the ml portfolio. If it fails -> the whole adapter switches to the stub
  (or, with REQUIRE_REAL_ML=1, the process exits). Then every other function is probed with a
  non-existent loan_id to find functions that still raise NotImplementedError.

Per-function status (shown by /api/health): "real" | "stub" | "not_ready" (NotImplementedError -> 503)
| "missing" (not defined in ml.api -> 503).

Optional ml features (both paths work): `sandbox_id` is passed to get_queue / get_loan / check_photo only
if the function's signature accepts it; `blurred_image` is taken from check_photo's result if present,
otherwise nothing is stored.
"""
from __future__ import annotations

import inspect
import logging
import os
import threading
from types import ModuleType
from typing import Any, Callable, Optional, Tuple

log = logging.getLogger("maqsad.ml")

FUNCTIONS = ("get_queue", "get_loan", "score_loan", "check_photo", "reset_sandbox", "get_color_thresholds",
             "score_scenario", "get_scenario_presets")
PROBE_LOAN_ID = "__maqsad_probe__"
PROBES = {  # called after a successful get_queue(); any error except NotImplementedError = implemented
    "get_loan": (PROBE_LOAN_ID,),
    "score_loan": (PROBE_LOAN_ID,),
    "check_photo": (PROBE_LOAN_ID, b""),
    "reset_sandbox": (PROBE_LOAN_ID,),
    "get_color_thresholds": (),  # read-only, cheap once the portfolio is built
    "get_scenario_presets": (),  # read-only
    "score_scenario": ({},),  # an empty scenario is rejected with ScenarioError: the function exists
}


class MLNotFound(Exception):
    """Unknown loan_id (ml raised KeyError/LookupError, e.g. LoanNotFoundError)."""


class MLBadInput(Exception):
    """ml rejected the input (ValueError)."""


class MLInvalidRequest(Exception):
    """ml rejected a request argument or scenario (InvalidRequestError / ScenarioError, Uzbek message) -> 400."""


class MLInvalidImage(Exception):
    """check_photo rejected the image (InvalidImageError, a ValueError)."""

    @property
    def too_large(self) -> bool:
        return "larger than" in str(self)


class MLNotReady(Exception):
    """The ml function is not implemented yet (NotImplementedError) or missing."""


class MLFailure(Exception):
    """Any other ml error."""


def _to_builtin(x: Any) -> Any:
    """numpy/pandas scalars and timestamps -> plain JSON-friendly Python (bytes are kept)."""
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


def _exit_process(message: str) -> None:
    log.critical("%s — exiting (REQUIRE_REAL_ML=1)", message)
    os._exit(3)


class MLAdapter:
    def __init__(self, impl: ModuleType, mode: str, require_real: bool = False,
                 on_fatal: Callable[[str], None] = _exit_process):
        self.require_real = require_real
        self._on_fatal = on_fatal
        self._lock = threading.Lock()
        self._warmup_done = threading.Event()
        self.warmup_error: Optional[str] = None
        self.clip_loaded = False  # from ml.api.warm_up(); False when CLIP is unavailable or warm_up() is missing
        self.clip_error: Optional[str] = None
        self._asset_files: Optional[frozenset] = None
        self._set_impl(impl, mode)
        if mode == "stub":
            self.warmup_state = "done"
            self._warmup_done.set()
        else:
            self.warmup_state = "pending"

    def _set_impl(self, impl: ModuleType, mode: str) -> None:
        self.impl, self.mode = impl, mode
        self.status = {
            f: ("stub" if mode == "stub" else "real") if callable(getattr(impl, f, None)) else "missing"
            for f in FUNCTIONS
        }
        self._takes_sandbox = {f: callable(getattr(impl, f, None)) and _accepts_kw(getattr(impl, f), "sandbox_id")
                               for f in ("get_queue", "get_loan", "check_photo")}

    @property
    def model_version(self) -> str:
        return str(getattr(self.impl, "MODEL_VERSION", None) or "unknown")

    # ---------- warm-up ----------

    def start_warmup(self) -> None:
        with self._lock:
            if self.warmup_state != "pending":
                return
            self.warmup_state = "running"
        threading.Thread(target=self._warmup, name="ml-warmup", daemon=True).start()

    def warmup(self) -> None:
        """Synchronous warm-up (tests, scripts)."""
        with self._lock:
            if self.warmup_state != "pending":
                return
            self.warmup_state = "running"
        self._warmup()

    def wait_warmup(self, timeout: Optional[float] = None) -> bool:
        return self._warmup_done.wait(timeout)

    def _warmup(self) -> None:
        self._ml_warm_up()
        try:
            n = len(self.impl.get_queue())
            log.info("ML warm-up: get_queue() ok (%d loans)", n)
        except Exception as e:  # noqa: BLE001 - any failure here means the real ML cannot serve the queue
            self.warmup_error = f"{type(e).__name__}: {e}"
            self.warmup_state = "failed"
            if self.require_real:
                self._warmup_done.set()
                self._on_fatal(f"ml.get_queue() failed during warm-up: {self.warmup_error}")
                return
            log.warning("ML warm-up failed (%s) -> switching to STUB", self.warmup_error)
            from . import ml_stub
            self._set_impl(ml_stub, "stub")
            self._warmup_done.set()
            return
        for name, args in PROBES.items():
            if self.status[name] != "real":
                continue
            try:
                getattr(self.impl, name)(*args)
            except NotImplementedError:
                self.status[name] = "not_ready"
                log.warning("ml.%s is not implemented yet -> 503 until it is", name)
            except Exception:  # noqa: BLE001 - expected (unknown loan / empty image): function exists
                pass
        self.warmup_state = "done"
        self._warmup_done.set()

    def _ml_warm_up(self) -> None:
        """ml.api.warm_up() (contract v1.5): portfolio + face detector + CLIP. Optional: get_queue() below still decides
        whether the real ML can serve; a failing warm_up() only leaves clip_loaded False."""
        fn = getattr(self.impl, "warm_up", None)
        if self.mode == "stub" or not callable(fn):
            return
        try:
            info = fn() or {}
            self.clip_loaded = bool(info.get("clip_loaded"))
            self.clip_error = info.get("clip_error")
            log.info("ML warm_up: %s", info)
        except Exception as e:  # noqa: BLE001
            self.clip_loaded, self.clip_error = False, f"{type(e).__name__}: {e}"
            log.warning("ml.warm_up() failed (%s); continuing with get_queue()", self.clip_error)

    def portfolio_asset_paths(self) -> frozenset:
        """Paths (relative to ml/assets, e.g. 'photos/x.jpg') of the photos in ml's photo index: the only portfolio
        images /api/asset serves. Empty in stub mode or if the index cannot be read."""
        if self._asset_files is None:
            files: frozenset = frozenset()
            if self.mode == "real":
                try:
                    from ml.photo.index import get_index
                    files = frozenset(f"photos/{im['file']}" for im in get_index().images)
                except Exception as e:  # noqa: BLE001
                    log.warning("photo index unavailable, /api/asset serves nothing: %r", e)
            self._asset_files = files
        return self._asset_files

    # ---------- calls ----------

    def _fn(self, name: str) -> Callable:
        if self.status[name] in ("missing", "not_ready"):
            raise MLNotReady(name)
        return getattr(self.impl, name)

    def _call(self, name: str, *args, **kwargs):
        fn = self._fn(name)
        try:
            return _to_builtin(fn(*args, **kwargs))
        except NotImplementedError as e:
            self.status[name] = "not_ready"
            raise MLNotReady(name) from e
        except LookupError as e:
            raise MLNotFound(str(e)) from e
        except ValueError as e:
            log.warning("ml.%s rejected input: %s", name, e)
            if type(e).__name__ in ("InvalidRequestError", "ScenarioError"):
                err = MLInvalidRequest(str(e))
                err.messages = getattr(e, "messages", None)          # {uz, en, ru} when ml provides them
                raise err from e
            if name == "check_photo":
                raise MLInvalidImage(str(e)) from e
            raise MLBadInput(str(e)) from e
        except Exception as e:  # noqa: BLE001 - map everything else to 502
            log.exception("ml.%s failed", name)
            raise MLFailure(f"{type(e).__name__}: {e}") from e

    def _sandbox_kw(self, name: str, sandbox_id: Optional[str]) -> dict:
        return {"sandbox_id": sandbox_id} if self._takes_sandbox.get(name) else {}

    def get_queue(self, sandbox_id: Optional[str] = None) -> list:
        return self._call("get_queue", **self._sandbox_kw("get_queue", sandbox_id))

    def get_loan(self, loan_id: str, sandbox_id: Optional[str] = None) -> dict:
        return self._call("get_loan", loan_id, **self._sandbox_kw("get_loan", sandbox_id))

    def score_loan(self, loan_id: str) -> dict:
        return self._call("score_loan", loan_id)

    def get_color_thresholds(self) -> dict:
        return self._call("get_color_thresholds")

    def score_scenario(self, scenario: dict) -> dict:
        return self._call("score_scenario", scenario)

    def get_scenario_presets(self) -> list:
        return self._call("get_scenario_presets")

    def check_photo(self, loan_id: str, image_bytes: bytes, lat=None, lon=None, captured_at=None,
                    sandbox_id: Optional[str] = None) -> Tuple[dict, Optional[bytes]]:
        """Returns (result without blurred_image, blurred_image bytes or None)."""
        kwargs = dict(lat=lat, lon=lon, captured_at=captured_at, **self._sandbox_kw("check_photo", sandbox_id))
        raw = dict(self._call("check_photo", loan_id, image_bytes, **kwargs))
        blurred = raw.pop("blurred_image", None)
        if blurred is not None and not isinstance(blurred, (bytes, bytearray)):
            log.warning("ml.check_photo blurred_image is %s, not bytes; ignoring", type(blurred).__name__)
            blurred = None
        return raw, (bytes(blurred) if blurred is not None else None)

    def reset_sandbox(self, sandbox_id: Optional[str]) -> None:
        """Best effort: a missing / not-ready reset_sandbox is logged, not raised."""
        try:
            self._call("reset_sandbox", sandbox_id)
        except (MLNotReady, MLFailure, MLNotFound, MLBadInput) as e:
            log.warning("ml.reset_sandbox(%r) skipped: %s", sandbox_id, e)


def load_adapter() -> MLAdapter:
    require_real = os.environ.get("REQUIRE_REAL_ML", "0") == "1"
    try:
        if os.environ.get("ML_FORCE_STUB", "0") == "1":
            raise ImportError("ML_FORCE_STUB=1")
        import ml.api as impl
        log.info("ML mode: real (ml.api imported, MODEL_VERSION=%s)", getattr(impl, "MODEL_VERSION", "?"))
        return MLAdapter(impl, "real", require_real=require_real)
    except Exception as e:  # noqa: BLE001 - ImportError, OSError (libgomp.so.1), SyntaxError, ...
        if require_real:
            raise RuntimeError(f"REQUIRE_REAL_ML=1 but ml.api could not be imported: {e!r}") from e
        log.warning("ML mode: STUB (ml.api not importable: %r)", e)
        from . import ml_stub
        return MLAdapter(ml_stub, "stub")
