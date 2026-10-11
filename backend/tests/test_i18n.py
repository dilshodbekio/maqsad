"""Accept-Language: error `detail` messages come back in uz / en / ru, never as English pydantic text."""
import copy
import re

import pytest

from backend.app import i18n
from backend.tests.conftest import png_bytes

CYR = re.compile(r"[А-Яа-яЁё]")
PYDANTIC_EN = re.compile(r"Field required|Input should|valid datetime|valid integer|Value error|String should|Extra inputs", re.I)
GOOD = {"purpose": "chorva", "amount_uzs": 50_000_000, "invoice_amount_uzs": 50_000_000, "region": None, "snapshot_day": 60,
        "supplier": {"mode": "new", "oked": "01.41", "registered_days_before": 900},
        "transactions": [{"day": 1, "type": "cash_withdrawal", "amount_uzs": 42_000_000}]}


def H(lang):
    return {"Accept-Language": lang}


@pytest.mark.parametrize("header,lang", [("ru", "ru"), ("ru-RU,ru;q=0.9,en;q=0.8", "ru"), ("en-US,en;q=0.9", "en"), ("uz", "uz"),
                                         ("de-DE", "uz"), ("", "uz"), (None, "uz"), ("fr, en;q=0.5", "en")])
def test_parse_lang(header, lang):
    assert i18n.parse_lang(header) == lang


def test_every_message_has_all_languages_and_matching_placeholders():
    for key, m in i18n.MESSAGES.items():
        assert set(m) == set(i18n.LANGS), key
        ph = {l: set(re.findall(r"\{(\w+)\}", t)) for l, t in m.items()}
        assert ph["uz"] == ph["en"] == ph["ru"], key
        assert all(t.strip() for t in m.values()), key
        assert CYR.search(m["ru"]) and not CYR.search(m["en"]) and not CYR.search(m["uz"]), key


def test_unknown_loan_404_in_three_languages(client):
    d = {l: client.get("/api/loan/L9999999", headers=H(l)) for l in ("uz", "en", "ru")}
    assert all(r.status_code == 404 and "L9999999" in r.json()["detail"] for r in d.values())
    assert d["uz"].json()["detail"].startswith("Kredit topilmadi")
    assert d["en"].json()["detail"].startswith("Loan not found")
    assert d["ru"].json()["detail"].startswith("Кредит не найден")
    assert client.get("/api/loan/L9999999").json()["detail"] == d["uz"].json()["detail"]     # default uz


@pytest.mark.parametrize("lang", ["uz", "en", "ru"])
def test_validation_422_is_localized_and_names_the_field(client, loan_id, lang):
    r = client.post("/api/check_photo", data={"loan_id": loan_id, "captured_at": "not-a-date"},
                    files={"image": ("a.png", png_bytes(), "image/png")}, headers=H(lang))
    d = r.json()["detail"]
    assert r.status_code == 422 and "captured_at" in d and not PYDANTIC_EN.search(d), d
    assert bool(CYR.search(d)) == (lang == "ru")
    r = client.post("/api/check_photo", data={"loan_id": ""}, files={"image": ("a.png", png_bytes(), "image/png")}, headers=H(lang))
    d = r.json()["detail"]
    assert r.status_code == 422 and "loan_id" in d and not PYDANTIC_EN.search(d), d


@pytest.mark.parametrize("lang", ["uz", "en", "ru"])
def test_queue_bad_query_422_localized(client, lang):
    r = client.get("/api/queue?limit=0&color=blue", headers=H(lang))
    d = r.json()["detail"]
    assert r.status_code == 422 and "limit" in d and "color" in d and not PYDANTIC_EN.search(d), d
    assert " or " not in d


@pytest.mark.parametrize("lang,start", [("uz", "Ssenariy noto'g'ri"), ("en", "The scenario is invalid"), ("ru", "Сценарий некорректен")])
def test_scenario_validation_localized(client, lang, start):
    bad = copy.deepcopy(GOOD)
    bad["purpose"], bad["snapshot_day"], bad["hacker"] = "kosmos", 1000, 1
    r = client.post("/api/score_scenario", json=bad, headers=H(lang))
    d = r.json()["detail"]
    assert r.status_code == 400 and d.startswith(start) and not PYDANTIC_EN.search(d), d
    assert " or " not in d and "purpose" in d and "snapshot_day" in d
    assert bool(CYR.search(d)) == (lang == "ru")


@pytest.mark.parametrize("lang,text", [("uz", "Faqat JPEG"), ("en", "Only JPEG"), ("ru", "Принимаются только")])
def test_bad_image_type_localized(client, loan_id, lang, text):
    r = client.post("/api/check_photo", data={"loan_id": loan_id}, files={"image": ("a.jpg", b"just text" * 20, "image/jpeg")}, headers=H(lang))
    assert r.status_code == 400 and r.json()["detail"].startswith(text)


@pytest.mark.parametrize("lang,text", [("uz", "8 MB"), ("en", "8 MB"), ("ru", "8 МБ")])
def test_too_large_localized(client, loan_id, lang, text):
    big = b"\xff\xd8\xff\xe0" + b"\0" * (9 * 1024 * 1024)
    r = client.post("/api/check_photo", data={"loan_id": loan_id}, files={"image": ("a.jpg", big, "image/jpeg")}, headers=H(lang))
    assert r.status_code == 413 and text in r.json()["detail"], r.text


@pytest.mark.parametrize("lang", ["uz", "en", "ru"])
def test_default_http_errors_localized(client, lang):
    r = client.get("/api/nope", headers=H(lang))
    assert r.status_code == 404 and r.json()["detail"] == i18n.MESSAGES["api_not_found"][lang]
    r = client.post("/api/admin/reset", headers=H(lang))
    assert r.status_code == 403 and r.json()["detail"] == i18n.MESSAGES["admin_forbidden"][lang]
    r = client.get("/api/draft_report/job/" + "0" * 32, headers=H(lang))
    assert r.status_code == 404 and r.json()["detail"] == i18n.MESSAGES["job_not_found"][lang]


@pytest.mark.parametrize("lang", ["uz", "en", "ru"])
def test_draft_report_without_llm_503_localized(client, loan_id, lang):
    r = client.post("/api/draft_report", json={"loan_id": loan_id}, headers=H(lang))
    assert r.status_code == 503 and r.json()["detail"] == i18n.MESSAGES["report_no_llm"][lang]


def test_draft_report_is_written_and_cached_per_language(make_client, monkeypatch):
    import json

    from backend.app import llm
    from backend.tests.conftest import draft

    c = make_client(llm_base_url="http://llm.invalid/v1", llm_model="test-model")
    loan_id = c.get("/api/queue").json()["items"][0]["loan_id"]
    calls = []

    def fake_chat(settings, messages):
        calls.append(messages)
        return json.dumps({"text": "Draft text.", "facts": []}), 10, 5

    monkeypatch.setattr(llm, "_chat", fake_chat)
    out = {}
    for lang in ("en", "ru", "uz", "en"):
        r = c.post("/api/draft_report", json={"loan_id": loan_id}, headers=H(lang))
        assert r.status_code in (200, 202), r.text
        if r.status_code == 202:
            import time
            job = r.json()["job_id"]
            for _ in range(100):
                j = c.get(f"/api/draft_report/job/{job}").json()
                if j["status"] == "done":
                    out[lang] = j["result"]
                    break
                time.sleep(0.1)
        else:
            out[lang] = r.json()
    assert [out[l]["lang"] for l in ("en", "ru", "uz")] == ["en", "ru", "uz"]
    assert len(calls) == 3, [m[0]["content"][:30] for m in calls]                                   # the 2nd English request came from the cache
    sys_prompts = [m[0]["content"] for m in calls]
    assert "Write the text in English" in sys_prompts[0] and "на русском языке" in sys_prompts[1] and "o'zbek tilida" in sys_prompts[2]
    payload = json.loads(calls[0][1]["content"])
    assert all("text_uz" not in r and isinstance(r["text"], str) for r in payload["reasons"])
