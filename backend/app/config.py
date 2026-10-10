"""Runtime settings, read from environment variables (see backend/.env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _float_or_none(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    return float(value)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    frontend_dist: Path
    ml_assets_dir: Path
    admin_token: str
    rate_limit_per_min: int
    rate_limit_ip_per_min: int
    max_upload_bytes: int
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout_s: float
    llm_price_in_per_mtok: Optional[float]  # USD per 1M input tokens (env LLM_PRICE_IN_PER_1M)
    llm_price_out_per_mtok: Optional[float]  # USD per 1M output tokens (env LLM_PRICE_OUT_PER_1M)
    llm_reasoning_effort: str  # sent as reasoning_effort; "" = not sent

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            data_dir=Path(env("DATA_DIR", "./data_runtime")).resolve(),
            frontend_dist=Path(env("FRONTEND_DIST", str(REPO_ROOT / "frontend" / "dist"))).resolve(),
            ml_assets_dir=Path(env("ML_ASSETS_DIR", str(REPO_ROOT / "ml" / "assets"))).resolve(),
            admin_token=env("ADMIN_TOKEN", ""),
            rate_limit_per_min=int(env("RATE_LIMIT_PER_MIN", "20")),
            rate_limit_ip_per_min=int(env("RATE_LIMIT_IP_PER_MIN", "120")),
            max_upload_bytes=int(float(env("MAX_UPLOAD_MB", "8")) * 1024 * 1024),
            llm_base_url=env("LLM_BASE_URL", "").rstrip("/"),
            llm_model=env("LLM_MODEL", ""),
            llm_api_key=env("LLM_API_KEY", ""),
            llm_timeout_s=float(env("LLM_TIMEOUT_S", "30")),
            llm_price_in_per_mtok=_float_or_none(env("LLM_PRICE_IN_PER_1M") or env("LLM_PRICE_IN_PER_MTOK")),
            llm_price_out_per_mtok=_float_or_none(env("LLM_PRICE_OUT_PER_1M") or env("LLM_PRICE_OUT_PER_MTOK")),
            llm_reasoning_effort=env("LLM_REASONING_EFFORT", "minimal").strip().strip("\"'").strip(),
        )
