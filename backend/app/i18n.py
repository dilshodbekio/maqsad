"""Backend message language (uz / en / ru): the caller's language comes from the `Accept-Language` header (default uz).

Every user-visible error text is a `Msg("key", **params)`: a `str` that already holds the Uzbek text (so code that treats it as a
plain string keeps working) and remembers its key and parameters, so the error handlers can render it in the caller's language.
"""
from __future__ import annotations

import re

LANGS = ("uz", "en", "ru")
DEFAULT_LANG = "uz"

MESSAGES: dict[str, dict[str, str]] = {
    "too_many": dict(uz="Juda ko'p so'rov. Bir daqiqadan keyin qayta urinib ko'ring.",
                     en="Too many requests. Please try again in a minute.",
                     ru="Слишком много запросов. Повторите попытку через минуту."),
    "bad_type": dict(uz="Faqat JPEG, PNG yoki WebP rasm qabul qilinadi.",
                     en="Only JPEG, PNG or WebP images are accepted.",
                     ru="Принимаются только изображения JPEG, PNG или WebP."),
    "empty_image": dict(uz="Rasm fayli bo'sh.", en="The image file is empty.", ru="Файл изображения пуст."),
    "no_image": dict(uz="Rasm yuborilmadi: 'image' maydoniga JPEG, PNG yoki WebP fayl biriktiring.",
                     en="No image was sent: attach a JPEG, PNG or WebP file to the 'image' field.",
                     ru="Изображение не отправлено: прикрепите файл JPEG, PNG или WebP к полю 'image'."),
    "report_not_ready": dict(uz="Dalolatnoma qoralamasi hozircha tayyor emas",
                             en="The inspection report draft is not ready yet",
                             ru="Черновик акта проверки пока не готов"),
    "report_no_llm": dict(uz="Dalolatnoma qoralamasi hozircha tayyor emas: LLM ulanmagan. Kredit sahifasi undan mustaqil ishlaydi.",
                          en="The inspection report draft is not ready yet: no LLM is connected. The loan page works without it.",
                          ru="Черновик акта проверки пока не готов: LLM не подключена. Страница кредита работает без неё."),
    "report_no_llm_short": dict(uz="Dalolatnoma qoralamasi hozircha tayyor emas: LLM ulanmagan.",
                                en="The inspection report draft is not ready yet: no LLM is connected.",
                                ru="Черновик акта проверки пока не готов: LLM не подключена."),
    "not_ready": dict(uz="Bu funksiya hali ulanmagan", en="This function is not connected yet", ru="Эта функция ещё не подключена"),
    "bad_image": dict(uz="Rasmni o'qib bo'lmadi: fayl buzilgan yoki rasm emas.",
                      en="The image could not be read: the file is corrupted or is not an image.",
                      ru="Не удалось прочитать изображение: файл повреждён или не является изображением."),
    "http_400": dict(uz="So'rov noto'g'ri.", en="Bad request.", ru="Некорректный запрос."),
    "http_403": dict(uz="Ruxsat yo'q.", en="Access denied.", ru="Доступ запрещён."),
    "http_404": dict(uz="Topilmadi.", en="Not found.", ru="Не найдено."),
    "http_405": dict(uz="Bu metod qo'llab-quvvatlanmaydi.", en="This method is not supported.", ru="Этот метод не поддерживается."),
    "http_413": dict(uz="So'rov hajmi juda katta.", en="The request is too large.", ru="Размер запроса слишком велик."),
    "http_429": dict(uz="Juda ko'p so'rov. Bir daqiqadan keyin qayta urinib ko'ring.",
                     en="Too many requests. Please try again in a minute.",
                     ru="Слишком много запросов. Повторите попытку через минуту."),
    "validation": dict(uz="So'rov ma'lumotlari noto'g'ri — {details}", en="The request data is invalid — {details}",
                       ru="Некорректные данные запроса — {details}"),
    "loan_not_found": dict(uz="Kredit topilmadi: {loan_id}", en="Loan not found: {loan_id}", ru="Кредит не найден: {loan_id}"),
    "ml_not_ready": dict(uz="Bu funksiya hali ulanmagan ({name}).", en="This function is not connected yet ({name}).",
                         ru="Эта функция ещё не подключена ({name})."),
    "too_large": dict(uz="Rasm hajmi {mb} MB dan oshmasligi kerak.", en="The image must not be larger than {mb} MB.",
                      ru="Размер изображения не должен превышать {mb} МБ."),
    "body_too_large": dict(uz="So'rov hajmi {kb} KB dan oshmasligi kerak.", en="The request must not be larger than {kb} KB.",
                           ru="Размер запроса не должен превышать {kb} КБ."),
    "request_rejected": dict(uz="So'rov qabul qilinmadi: {detail}", en="The request was rejected: {detail}",
                             ru="Запрос не принят: {detail}"),
    "model_rejected": dict(uz="Model kiritilgan ma'lumotni qabul qilmadi: {detail}", en="The model did not accept the input: {detail}",
                           ru="Модель не приняла введённые данные: {detail}"),
    "ml_failure": dict(uz="Model xizmatida xatolik yuz berdi, keyinroq qayta urinib ko'ring.",
                       en="The model service failed. Please try again later.",
                       ru="Произошла ошибка в сервисе модели. Повторите попытку позже."),
    "internal": dict(uz="Ichki xatolik yuz berdi.", en="An internal error occurred.", ru="Произошла внутренняя ошибка."),
    "json_required": dict(uz="So'rov JSON ko'rinishida bo'lishi kerak.", en="The request must be JSON.", ru="Запрос должен быть в формате JSON."),
    "scenario_invalid": dict(uz="Ssenariy noto'g'ri: {details}", en="The scenario is invalid: {details}", ru="Сценарий некорректен: {details}"),
    "json_malformed": dict(uz="JSON noto'g'ri formatda.", en="The JSON is malformed.", ru="JSON имеет неверный формат."),
    "base64_invalid": dict(uz="image_base64 noto'g'ri (base64 emas).", en="image_base64 is invalid (not base64).",
                           ru="image_base64 некорректно (это не base64)."),
    "content_type": dict(uz="So'rov multipart/form-data (asosiy) yoki application/json bo'lishi kerak.",
                         en="The request must be multipart/form-data (preferred) or application/json.",
                         ru="Запрос должен быть multipart/form-data (основной вариант) или application/json."),
    "photo_not_found": dict(uz="Rasm topilmadi.", en="Photo not found.", ru="Фото не найдено."),
    "regen_limit": dict(uz="Qayta yaratish daqiqada bir marta mumkin. Biroz kuting.",
                        en="A report can be regenerated once a minute. Please wait a little.",
                        ru="Пересоздавать акт можно раз в минуту. Немного подождите."),
    "busy": dict(uz="Dalolatnoma xizmati hozir band, birozdan keyin qayta urinib ko'ring.",
                 en="The report service is busy right now. Please try again shortly.",
                 ru="Сервис актов сейчас занят, повторите попытку чуть позже."),
    "draft_failed": dict(uz="Dalolatnoma qoralamasi hozircha tayyor emas: LLM xatosi. Keyinroq qayta urinib ko'ring.",
                         en="The inspection report draft is not ready: LLM error. Please try again later.",
                         ru="Черновик акта пока не готов: ошибка LLM. Повторите попытку позже."),
    "job_not_found": dict(uz="Ish topilmadi. Dalolatnomani qayta so'rang.", en="Job not found. Please request the report again.",
                          ru="Задание не найдено. Запросите акт заново."),
    "admin_forbidden": dict(uz="Ruxsat yo'q: X-Admin-Token noto'g'ri yoki ADMIN_TOKEN sozlanmagan.",
                            en="Access denied: X-Admin-Token is wrong or ADMIN_TOKEN is not configured.",
                            ru="Доступ запрещён: X-Admin-Token неверен или ADMIN_TOKEN не настроен."),
    "file_not_found": dict(uz="Fayl topilmadi.", en="File not found.", ru="Файл не найден."),
    "api_not_found": dict(uz="API endpoint topilmadi.", en="API endpoint not found.", ru="Эндпоинт API не найден."),
    # pieces of validation messages
    "v_missing": dict(uz="majburiy maydon yetishmaydi", en="this field is required", ru="обязательное поле отсутствует"),
    "v_extra": dict(uz="noma'lum kalit", en="unknown key", ru="неизвестный ключ"),
    "v_object": dict(uz="obyekt bo'lishi kerak", en="must be an object", ru="должно быть объектом"),
    "v_list": dict(uz="ro'yxat bo'lishi kerak", en="must be a list", ru="должно быть списком"),
    "v_int": dict(uz="butun son bo'lishi kerak", en="must be a whole number", ru="должно быть целым числом"),
    "v_number": dict(uz="son bo'lishi kerak", en="must be a number", ru="должно быть числом"),
    "v_string": dict(uz="matn bo'lishi kerak", en="must be text", ru="должно быть строкой"),
    "v_range": dict(uz="qiymat ruxsat etilgan oralig'dan tashqarida", en="the value is outside the allowed range",
                    ru="значение вне допустимого диапазона"),
    "v_too_long": dict(uz="{n} tadan oshmasligi kerak", en="must not exceed {n}", ru="не должно превышать {n}"),
    "v_too_short": dict(uz="juda qisqa (kamida {n})", en="is too short (at least {n})", ru="слишком короткое (не менее {n})"),
    "v_literal": dict(uz="ruxsat etilgan qiymatlardan biri bo'lishi kerak ({expected})", en="must be one of the allowed values ({expected})",
                      ru="должно быть одним из допустимых значений ({expected})"),
    "v_union": dict(uz="ta'minotchi 'mode' qiymati 'existing' yoki 'new' bo'lishi kerak",
                    en="the supplier 'mode' must be 'existing' or 'new'", ru="'mode' поставщика должен быть 'existing' или 'new'"),
    "v_format": dict(uz="format noto'g'ri", en="the format is invalid", ru="неверный формат"),
    "v_datetime": dict(uz="sana va vaqt noto'g'ri formatda (masalan 2026-05-01T10:30:00Z)",
                       en="the date and time format is invalid (for example 2026-05-01T10:30:00Z)",
                       ru="неверный формат даты и времени (например, 2026-05-01T10:30:00Z)"),
    "v_bool": dict(uz="ha/yo'q qiymati bo'lishi kerak", en="must be true or false", ru="должно быть true или false"),
    "v_json": dict(uz="JSON noto'g'ri formatda", en="the JSON is malformed", ru="JSON имеет неверный формат"),
    "v_invalid": dict(uz="qiymat noto'g'ri", en="the value is invalid", ru="некорректное значение"),
    "v_body": dict(uz="so'rov tanasi", en="request body", ru="тело запроса"),
    "v_scenario": dict(uz="ssenariy", en="scenario", ru="сценарий"),
}

_UZ_TO_KEY = {v["uz"]: k for k, v in MESSAGES.items() if "{" not in v["uz"]}


def _resolve(params: dict, lang: str) -> dict:
    """A parameter may be a function of the language (e.g. a detail text that is itself translated)."""
    return {k: (v(lang) if callable(v) else v) for k, v in params.items()}


class Msg(str):
    """The Uzbek text of message `key` (parameters filled in) that also remembers how to say it in another language."""
    key: str
    params: dict

    def __new__(cls, key: str, **params):
        self = super().__new__(cls, MESSAGES[key]["uz"].format(**_resolve(params, "uz")))
        self.key, self.params = key, params
        return self

    def render(self, lang: str) -> str:
        return MESSAGES[self.key].get(lang, MESSAGES[self.key]["uz"]).format(**_resolve(self.params, lang))


def parse_lang(header: str | None) -> str:
    """First supported language of an Accept-Language header ('ru-RU,ru;q=0.9,en;q=0.8' -> 'ru'); default uz."""
    for part in (header or "").split(","):
        code = part.split(";")[0].strip().lower()[:2]
        if code in LANGS:
            return code
    return DEFAULT_LANG


def render(text, lang: str) -> str:
    """A Msg is rendered in `lang`; a plain string that equals a known parameter-free Uzbek message is looked up; anything else is returned as is."""
    if isinstance(text, Msg):
        return text.render(lang)
    key = _UZ_TO_KEY.get(text) if isinstance(text, str) else None
    return MESSAGES[key].get(lang, text) if key else text


def pick(messages: dict | None, lang: str, fallback: str) -> str:
    """A {uz,en,ru} dict from ml (error messages) in `lang`; falls back to the Uzbek text."""
    if isinstance(messages, dict):
        return messages.get(lang) or messages.get("uz") or fallback
    return fallback


def _m(key: str, lang: str, **kw) -> str:
    return MESSAGES[key].get(lang, MESSAGES[key]["uz"]).format(**kw)


def error_phrase(err: dict, lang: str) -> str:
    """One pydantic error -> a phrase in `lang` (no English pydantic text reaches the user)."""
    t, ctx = err.get("type", ""), err.get("ctx") or {}
    if t == "extra_forbidden":
        return _m("v_extra", lang)
    if t == "missing":
        return _m("v_missing", lang)
    if t in ("model_type", "dict_type", "model_attributes_type"):
        return _m("v_object", lang)
    if t in ("list_type", "tuple_type", "set_type"):
        return _m("v_list", lang)
    if t in ("int_type", "int_parsing", "int_from_float", "int_parsing_size"):
        return _m("v_int", lang)
    if t in ("float_type", "float_parsing", "finite_number", "decimal_parsing"):
        return _m("v_number", lang)
    if t in ("string_type", "bytes_type"):
        return _m("v_string", lang)
    if t in ("greater_than_equal", "less_than_equal", "greater_than", "less_than", "multiple_of"):
        return _m("v_range", lang)
    if t in ("too_long", "string_too_long"):
        return _m("v_too_long", lang, n=ctx.get("max_length", ""))
    if t in ("too_short", "string_too_short"):
        return _m("v_too_short", lang, n=ctx.get("min_length", ""))
    if t in ("literal_error", "enum"):
        return _m("v_literal", lang, expected=str(ctx.get("expected", "")).replace(" or ", ", "))
    if t in ("union_tag_invalid", "union_tag_not_found"):
        return _m("v_union", lang)
    if t == "string_pattern_mismatch":
        return _m("v_format", lang)
    if t.startswith(("datetime", "date_", "time_")) or t in ("date_type", "datetime_type"):
        return _m("v_datetime", lang)
    if t in ("bool_type", "bool_parsing"):
        return _m("v_bool", lang)
    if t == "json_invalid":
        return _m("v_json", lang)
    if t == "value_error":
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        return msg if msg and not re.search(r"[A-Za-z]{4,} [a-z]{2,}", msg) else _m("v_invalid", lang)
    return _m("v_invalid", lang)


def error_details(errors: list[dict], lang: str, limit: int = 5, root: str = "v_body") -> str:
    parts = []
    for err in errors[:limit]:
        loc = ".".join(str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path"))
        parts.append(f"{loc or _m(root, lang)}: {error_phrase(err, lang)}")
    return "; ".join(parts)
