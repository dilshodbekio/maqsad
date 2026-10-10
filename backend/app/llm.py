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


class LLMNotConfigured(Exception):
    pass


class LLMError(Exception):
    pass


SYSTEM_PROMPT = (
    "Siz bank kredit monitoringi xodimi uchun tekshiruv dalolatnomasining QORALAMASINI yozasiz. "
    "Faqat berilgan dalillar (evidence) va sabablar (reasons) dagi faktlardan foydalaning, hech narsa "
    "o'ylab topmang. Hech kimni aybdor demang: 'xavf belgisi', 'birinchi navbatda tekshirish tavsiya etiladi' "
    "kabi iboralarni ishlating. Yakuniy qarorni xodim qabul qiladi. Matn o'zbek tilida (lotin). "
    "Faqat JSON qaytaring: {\"text_uz\": string, \"facts\": [{\"claim\": string, \"evidence_ref\": string}]}. "
    "Har bir fakt bitta dalilga tayansin; evidence_ref aynan dalildagi 'ref' qiymati bo'lsin; "
    "raqamlarni dalildagidek yozing."
)

_NUM_RE = re.compile(r"\d+(?:[   .,]\d+)*")


def _numbers(text: str) -> set:
    """Digit sequences with thousand/decimal separators removed: '12 500 000' -> '12500000'."""
    return {re.sub(r"\D", "", m.group()) for m in _NUM_RE.finditer(text)}


def fact_check(facts: list, evidence: list) -> list:
    by_ref = {e.get("ref"): e for e in evidence if e.get("ref")}
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


def _chat(settings: Settings, messages: list) -> tuple:
    """Returns (content, input_tokens, output_tokens)."""
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    body = {"model": settings.llm_model, "messages": messages, "temperature": 0.1,
            "response_format": {"type": "json_object"}}
    r = httpx.post(f"{settings.llm_base_url}/chat/completions", json=body, headers=headers,
                   timeout=settings.llm_timeout_s)
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage") or {}
    return data["choices"][0]["message"]["content"], usage.get("prompt_tokens"), usage.get("completion_tokens")


def draft_report(settings: Settings, loan_id: str, evidence: list, reasons: list, db=None) -> dict:
    if not settings.llm_configured:
        raise LLMNotConfigured()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"reasons": reasons, "evidence": evidence}, ensure_ascii=False)},
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
    }
