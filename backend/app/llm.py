"""LLM draft report: OpenAI-compatible chat API (prototype: external; production: local vLLM/Ollama,
only LLM_BASE_URL changes). The prompt gets ONLY this loan's evidence[] and reasons[].
Every fact is then checked in code (no LLM): evidence_ref must exist and every number in the
claim must appear in the evidence. Failed facts are kept and marked unverified=True.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import httpx

from .config import Settings

log = logging.getLogger("maqsad.llm")


PROMPT_VERSION = "2"  # part of the draft cache key: a prompt change invalidates cached drafts


class LLMNotConfigured(Exception):
    pass


class LLMError(Exception):
    pass


class LLMRateLimited(LLMError):
    """HTTP 429 from the LLM API: the draft queue keeps the job and retries after `retry_after_s`."""

    def __init__(self, message: str, retry_after_s: float):
        super().__init__(message)
        self.retry_after_s = retry_after_s


SYSTEM_PROMPT = (
    "Siz bank kredit monitoringi xodimi uchun tekshiruv dalolatnomasining QORALAMASINI yozasiz. "
    "Faqat berilgan dalillar (evidence) va sabablar (reasons) dagi faktlardan foydalaning, hech narsa "
    "o'ylab topmang. Hech kimni aybdor demang: 'xavf belgisi', 'birinchi navbatda tekshirish tavsiya etiladi' "
    "kabi iboralarni ishlating. Yakuniy qarorni xodim qabul qiladi. Matn o'zbek tilida (lotin). "
    "Faqat JSON qaytaring: {\"text_uz\": string, \"facts\": [{\"claim\": string, \"evidence_ref\": string}]}. "
    "Har bir fakt bitta dalilga tayansin; evidence_ref FAQAT 'allowed_refs' ro'yxatidagi qiymatlardan biri "
    "bo'lsin (boshqa qiymat tekshiruvdan o'tmaydi); raqamlarni dalildagidek yozing. "
    "'risk' ehtimollik emas, foiz sifatida yozmang."
)

_NUM_RE = re.compile(r"\d+(?:[   .,]\d+)*")


def _numbers(text: str) -> set:
    """Digit sequences with thousand/decimal separators removed: '12 500 000' -> '12500000'."""
    return {re.sub(r"\D", "", m.group()) for m in _NUM_RE.finditer(text)}


def fact_check(facts: list, evidence: list) -> list:
    """evidence_ref must be one of evidence[].ref (nothing else, e.g. not a reasons[].evidence_refs value
    that has no evidence item); every number in the claim must appear in that evidence item."""
    by_ref = {e["ref"]: e for e in evidence if isinstance(e.get("ref"), str) and e["ref"]}
    checked = []
    for f in facts:
        claim = str(f.get("claim", "")).strip()
        ref = str(f.get("evidence_ref", "")).strip()
        if not claim:
            continue
        ok = ref in by_ref
        if ok:
            # every number in the claim must appear in the cited evidence item
            ev_numbers = _numbers(json.dumps(by_ref[ref], ensure_ascii=False))
            ok = _numbers(claim) <= ev_numbers
        checked.append({"claim": claim, "evidence_ref": ref, "unverified": not ok})
    return checked


def _cost(settings: Settings, tin: Optional[int], tout: Optional[int]) -> Optional[float]:
    if settings.llm_price_in_per_mtok is None or settings.llm_price_out_per_mtok is None:
        return None  # price unknown -> never guessed
    if tin is None or tout is None:
        return None
    return round(tin / 1e6 * settings.llm_price_in_per_mtok + tout / 1e6 * settings.llm_price_out_per_mtok, 6)


_RETRY_STATUS = {500, 502, 503, 504}  # transient (Gemini: 503 "high demand"); 429 goes back to the draft queue
DEFAULT_RETRY_AFTER_S = 15.0
_RETRY_IN_RE = re.compile(r"retry in ((?:[0-9]+h)?(?:[0-9]+m)?(?:[0-9]+(?:\.[0-9]+)?s)?)", re.I)
_DURATION_RE = re.compile(r"(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+(?:\.[0-9]+)?)s)?$")


def _seconds(text: str) -> Optional[float]:
    """'2.1s' / '6549s' / '1h49m9.55s' -> seconds."""
    m = _DURATION_RE.match(text or "")
    if not m or not any(m.groups()):
        return None
    h, mi, sec = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(sec or 0)


def _retry_after(r) -> float:
    """Retry-After header, else Gemini's 'Please retry in 2.1s' / '1h49m9.5s' text, else a default."""
    try:
        return max(0.0, float(r.headers.get("retry-after", "")))
    except (TypeError, ValueError):
        pass
    m = _RETRY_IN_RE.search(r.text or "")
    parsed = _seconds(m.group(1)) if m else None
    return parsed if parsed is not None else DEFAULT_RETRY_AFTER_S
_RETRY_DELAYS_S = (1.0, 3.0)


def _clean(value: str) -> str:
    """Env values pasted into a dashboard often keep whitespace/newlines or quotes; Gemini then answers 400
    ('Please pass a valid API key' / 'unexpected model name format')."""
    return (value or "").strip().strip("\"'").strip()


def _chat(settings: Settings, messages: list) -> tuple:
    """Returns (content, input_tokens, output_tokens)."""
    api_key, model, base_url = _clean(settings.llm_api_key), _clean(settings.llm_model), _clean(settings.llm_base_url)

    def scrub(text: str) -> str:
        text = " ".join(str(text).split())[:300]
        return text.replace(api_key, "***") if api_key else text

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "messages": messages, "temperature": 0.1,
            "response_format": {"type": "json_object"}}
    if settings.llm_reasoning_effort:
        body["reasoning_effort"] = settings.llm_reasoning_effort  # lowest thinking: faster and cheaper
    for attempt in range(len(_RETRY_DELAYS_S) + 1):
        try:
            r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body, headers=headers,
                           timeout=settings.llm_timeout_s)
        except httpx.HTTPError as e:  # transport errors may echo request headers -> scrub the key
            raise LLMError(f"{type(e).__name__}: {scrub(e)}") from None
        if r.status_code in _RETRY_STATUS and attempt < len(_RETRY_DELAYS_S):
            log.warning("LLM HTTP %s (attempt %s), retrying: %s", r.status_code, attempt + 1, scrub(r.text))
            time.sleep(_RETRY_DELAYS_S[attempt])
            continue
        if r.status_code == 429:
            raise LLMRateLimited(f"HTTP 429 from LLM API: {scrub(r.text)}", _retry_after(r))
        if r.is_error:
            # short response body (key scrubbed) so the 4xx/5xx cause is visible in the log
            raise LLMError(f"HTTP {r.status_code} from LLM API: {scrub(r.text)}")
        break
    data = r.json()
    usage = data.get("usage") or {}
    return data["choices"][0]["message"]["content"], usage.get("prompt_tokens"), usage.get("completion_tokens")


def draft_report(settings: Settings, loan_id: str, evidence: list, reasons: list, db=None) -> dict:
    if not settings.llm_configured:
        raise LLMNotConfigured()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "allowed_refs": [e["ref"] for e in evidence if e.get("ref")],
            "reasons": reasons,
            "evidence": evidence,
        }, ensure_ascii=False)},
    ]
    t = time.perf_counter()
    tin = tout = None
    try:
        content, tin, tout = _chat(settings, messages)
        parsed = json.loads(content)
        text_uz = str(parsed["text_uz"])
        facts = parsed.get("facts") or []
        if not isinstance(facts, list):
            raise ValueError("facts is not a list")
    except LLMRateLimited as e:
        log.warning("LLM rate limited loan=%s model=%s (retry in %ss)", loan_id, settings.llm_model, e.retry_after_s)
        raise
    except Exception as e:  # noqa: BLE001
        latency = round((time.perf_counter() - t) * 1000, 1)
        log.warning("LLM call failed loan=%s model=%s latency_ms=%s: %s", loan_id, settings.llm_model, latency, e)
        if db is not None:
            db.llm_log(loan_id=loan_id, model=settings.llm_model, latency_ms=latency, input_tokens=tin,
                       output_tokens=tout, cost_usd=_cost(settings, tin, tout), ok=False,
                       error=f"{type(e).__name__}: {e}"[:500])
        raise LLMError(str(e)) from e

    latency = round((time.perf_counter() - t) * 1000, 1)
    cost = _cost(settings, tin, tout)
    log.info("LLM call loan=%s model=%s latency_ms=%s in=%s out=%s cost_usd=%s",
             loan_id, settings.llm_model, latency, tin, tout, cost)
    if db is not None:
        db.llm_log(loan_id=loan_id, model=settings.llm_model, latency_ms=latency, input_tokens=tin,
                   output_tokens=tout, cost_usd=cost, ok=True)
    return {
        "text_uz": text_uz,
        "facts": fact_check([f for f in facts if isinstance(f, dict)], evidence),
        "model": settings.llm_model,
        "latency_ms": latency,
        "cost_usd": cost,
        "input_tokens": tin,
        "output_tokens": tout,
    }
