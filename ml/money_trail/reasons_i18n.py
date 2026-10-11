"""English and Russian wording of the money-trail reasons and evidence summaries (the Uzbek originals live in reasons.py).

Same facts and the same numbers in every language: only the sentence changes (digits, dates and codes are identical, which
`ml/eval/audit_texts.py` checks). Wording states facts about the money trail and never says that a person is guilty.
Money is "UZS" / "сум"; days are "days" / "дн." (no plural forms to get wrong).
"""
from __future__ import annotations

from .features import NEVER

LOW_PAY_SHARE, LOW_SUPPLIER_AGE_DAYS, LOW_ACTIVE_SPAN_DAYS = 0.25, 60, 100
MORE_THAN_100 = {"en": "more than 100%", "ru": "более 100%"}


def _p(v, lang: str) -> str:
    """'78%' ; a share above 100% is not shown as a percentage ('more than 100%')."""
    v = max(float(v), 0.0)
    return MORE_THAN_100[lang] if v > 1.0 else f"{v * 100:.0f}%"


def _d(v) -> str:
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"


def _en(code: str, v: float, c: dict) -> str:
    p = lambda x=v: _p(x, "en")
    if code == "first_cash_days":
        return ("No cash withdrawal since the loan was disbursed" if v >= NEVER else
                "The first cash withdrawal was on the day the loan was disbursed" if v < 0.5 else
                f"The first cash withdrawal was {_d(v)} days after disbursement")
    if code == "cash_share_d1":
        return f"{p()} of the funds were withdrawn in cash within 24 hours of disbursement"
    if code == "cash_share_d7":
        return f"{p()} of the funds were withdrawn in cash within 7 days of disbursement"
    if code == "cash_share_d30":
        return f"{p()} of the funds were withdrawn in cash within 30 days of disbursement"
    if code == "cash_share_all":
        return f"{p()} of the funds have been withdrawn in cash since disbursement"
    if code == "sup_pay_share":
        return ("No payment was made to the supplier" if v <= 0 else
                "Less than 1% of the loan amount was paid to the supplier" if v < 0.005 else
                f"Only {p()} of the loan amount was paid to the supplier" if v < LOW_PAY_SHARE else
                f"{p()} of the loan amount was paid to the supplier")
    if code == "first_sup_pay_days":
        return ("No payment to the supplier yet" if v >= NEVER else
                "The first payment to the supplier was made on the day the loan was disbursed" if v < 0.5 else
                f"The first payment to the supplier was made {_d(v)} days after disbursement")
    if code == "invoice_gap":
        return "No invoice on file" if v < 0 else f"{p()} of the invoice amount was not paid to the supplier"
    if code == "leave_share_30":
        return f"{p()} of the funds left the bank in the first 30 days (cash, another account or a person)"
    if code == "p2p_out_count":
        return f"Transfers to individuals: {v:.0f}"
    if code == "p2p_in_count":
        return f"Transfers received from individuals: {v:.0f}"
    if code == "big_inbound_share":
        return f"One large inflow equal to {p()} of the loan amount was received"
    if code == "big_inbound_day":
        return ("No large inflow recorded" if v < 0 else
                "The largest inflow arrived on the day the loan was disbursed" if v < 0.5 else
                f"The largest inflow arrived {_d(v)} days after disbursement")
    if code == "big_inbound_person":
        return "The largest inflow came from an individual" if v >= 0.5 else "The largest inflow came from an organisation"
    if code == "post_inbound_exit":
        return f"{p()} of the large inflow left the account within 5 days"
    if code == "income_rate_per_month":
        return ("No income expected yet (the loan was issued recently)" if v < 0 else
                "No income recorded since the loan was issued" if v == 0 else
                f"On average {v:.1f} income payments per month during the loan period")
    if code == "income_missing_flag":
        return "The loan was issued more than 90 days ago, but no income has been recorded yet"
    if code == "income_after_big_count":
        return "No income recorded after the large inflow" if v <= 0 else f"Income payments after the large inflow: {v:.0f}"
    if code == "income_gap_days":
        return f"{_d(v)} days have passed since the last income (or since income was expected)"
    if code == "income_stopped_flag":
        return "Income stopped after the large inflow"
    if code == "sup_in_degree":
        return f"This supplier appears on the invoices of {v:.0f} loans"
    if code == "sup_cash_share":
        return f"The supplier withdrew {p()} of the funds received in cash"
    if code == "sup_exit_share":
        return f"The supplier moved {p()} of the funds received out of its account"
    if code == "sup_cash_lag_days":
        return ("The supplier has not withdrawn cash" if v >= NEVER else
                "The supplier withdrew cash on the day it received the money" if v < 0.5 else
                f"The supplier withdrew cash {_d(v)} days after receiving the money")
    if code == "sup_active_span_days":
        return ("No activity recorded on the supplier's account yet" if v < 0 else
                f"The supplier's account has been active for only {_d(v)} days" if v < LOW_ACTIVE_SPAN_DAYS else
                f"The supplier's account has been active for {_d(v)} days")
    if code == "sup_age_at_first_payment_days":
        return ("The supplier has not received a payment yet" if v < 0 else
                "The supplier was registered on the day of the first payment" if v < 0.5 else
                f"The supplier was registered only {_d(v)} days before the first payment" if v < LOW_SUPPLIER_AGE_DAYS else
                f"The supplier was registered {_d(v)} days before the first payment")
    if code == "sup_employee_link":
        return "The supplier is linked to a bank employee"
    if code == "emp_branch_match":
        return "The employee linked to the supplier works at the branch that issued the loan"
    if code == "oked_mismatch":
        return f"The supplier's line of business (OKED {c['supplier_oked']}) does not match the loan purpose"
    raise KeyError(code)


def _ru(code: str, v: float, c: dict) -> str:
    p = lambda x=v: _p(x, "ru")
    if code == "first_cash_days":
        return ("С момента выдачи кредита наличные не снимались" if v >= NEVER else
                "Первое снятие наличных было в день выдачи кредита" if v < 0.5 else
                f"Первое снятие наличных было через {_d(v)} дн. после выдачи кредита")
    if code == "cash_share_d1":
        return f"В течение 24 часов после выдачи кредита снято наличными {p()} средств"
    if code == "cash_share_d7":
        return f"В течение 7 дней после выдачи кредита снято наличными {p()} средств"
    if code == "cash_share_d30":
        return f"В течение 30 дней после выдачи кредита снято наличными {p()} средств"
    if code == "cash_share_all":
        return f"С момента выдачи кредита снято наличными {p()} средств"
    if code == "sup_pay_share":
        return ("Платёж поставщику не переводился" if v <= 0 else
                "Поставщику переведено менее 1% суммы кредита" if v < 0.005 else
                f"Поставщику переведено лишь {p()} суммы кредита" if v < LOW_PAY_SHARE else
                f"Поставщику переведено {p()} суммы кредита")
    if code == "first_sup_pay_days":
        return ("Поставщику ещё не переводились платежи" if v >= NEVER else
                "Первый платёж поставщику сделан в день выдачи кредита" if v < 0.5 else
                f"Первый платёж поставщику сделан через {_d(v)} дн. после выдачи кредита")
    if code == "invoice_gap":
        return "Счёт-фактура отсутствует" if v < 0 else f"{p()} суммы счёта-фактуры не переведено поставщику"
    if code == "leave_share_30":
        return f"За первые 30 дней из банка выведено {p()} средств (наличные, другой счёт или физлицо)"
    if code == "p2p_out_count":
        return f"Переводов физическим лицам: {v:.0f}"
    if code == "p2p_in_count":
        return f"Получено переводов от физических лиц: {v:.0f}"
    if code == "big_inbound_share":
        return f"Поступил один крупный платёж, равный {p()} суммы кредита"
    if code == "big_inbound_day":
        return ("Крупных поступлений не зафиксировано" if v < 0 else
                "Самое крупное поступление пришло в день выдачи кредита" if v < 0.5 else
                f"Самое крупное поступление пришло через {_d(v)} дн. после выдачи кредита")
    if code == "big_inbound_person":
        return "Самое крупное поступление пришло от физического лица" if v >= 0.5 else "Самое крупное поступление пришло от организации"
    if code == "post_inbound_exit":
        return f"В течение 5 дней после крупного поступления выведено {p()} от него"
    if code == "income_rate_per_month":
        return ("Поступлений дохода пока не ожидалось (кредит выдан недавно)" if v < 0 else
                "С момента выдачи кредита поступлений дохода не зафиксировано" if v == 0 else
                f"В период кредита в среднем {v:.1f} поступлений дохода в месяц")
    if code == "income_missing_flag":
        return "Кредит выдан более 90 дней назад, но поступлений дохода пока нет"
    if code == "income_after_big_count":
        return "После крупного поступления доход не зафиксирован" if v <= 0 else f"Поступлений дохода после крупного платежа: {v:.0f}"
    if code == "income_gap_days":
        return f"С последнего поступления дохода (или с даты, когда его ждали) прошло {_d(v)} дн."
    if code == "income_stopped_flag":
        return "После крупного поступления доходы прекратились"
    if code == "sup_in_degree":
        return f"Этот поставщик указан в счетах-фактурах по {v:.0f} кредитам"
    if code == "sup_cash_share":
        return f"Поставщик снял наличными {p()} полученных средств"
    if code == "sup_exit_share":
        return f"Поставщик вывел со счёта {p()} полученных средств"
    if code == "sup_cash_lag_days":
        return ("Поставщик наличные не снимал" if v >= NEVER else
                "Поставщик снял наличные в день получения денег" if v < 0.5 else
                f"Поставщик снял наличные через {_d(v)} дн. после получения денег")
    if code == "sup_active_span_days":
        return ("Движения по счёту поставщика пока не зафиксированы" if v < 0 else
                f"Счёт поставщика был активен лишь {_d(v)} дн." if v < LOW_ACTIVE_SPAN_DAYS else
                f"Счёт поставщика был активен {_d(v)} дн.")
    if code == "sup_age_at_first_payment_days":
        return ("Поставщику ещё не поступали платежи" if v < 0 else
                "Поставщик зарегистрирован в день первого платежа" if v < 0.5 else
                f"Поставщик зарегистрирован лишь за {_d(v)} дн. до первого платежа" if v < LOW_SUPPLIER_AGE_DAYS else
                f"Поставщик зарегистрирован за {_d(v)} дн. до первого платежа")
    if code == "sup_employee_link":
        return "Поставщик связан с сотрудником банка"
    if code == "emp_branch_match":
        return "Сотрудник, связанный с поставщиком, работает в филиале, выдавшем кредит"
    if code == "oked_mismatch":
        return f"Вид деятельности поставщика (ОКЭД {c['supplier_oked']}) не соответствует цели кредита"
    raise KeyError(code)


def _cap(f):
    """A sentence starts with a capital letter ('more than 100% ...' -> 'More than 100% ...')."""
    def g(code: str, v: float, c: dict) -> str:
        s = f(code, v, c)
        return s[:1].upper() + s[1:]
    return g


FORMATTERS = {"en": _cap(_en), "ru": _cap(_ru)}

NO_REASON = {"uz": "Sezilarli xavf belgisi aniqlanmadi", "en": "No significant risk signal was found",
             "ru": "Существенных признаков риска не обнаружено"}

SOM = {"uz": "so'm", "en": "UZS", "ru": "сум"}

TXN_TYPE = {
    "uz": {"disbursement": "kredit tushgan", "cash_withdrawal": "naqd yechilgan", "supplier_payment": "ta'minotchiga o'tkazilgan",
           "p2p_out": "shaxsga o'tkazilgan", "p2p_in": "shaxsdan qabul qilingan", "external_transfer": "boshqa bankka o'tkazilgan",
           "income": "tushum qayd etilgan"},
    "en": {"disbursement": "loan received", "cash_withdrawal": "cash withdrawn", "supplier_payment": "transferred to the supplier",
           "p2p_out": "transferred to an individual", "p2p_in": "received from an individual",
           "external_transfer": "transferred to another bank", "income": "income recorded"},
    "ru": {"disbursement": "кредит получен", "cash_withdrawal": "снято наличными", "supplier_payment": "переведено поставщику",
           "p2p_out": "переведено физлицу", "p2p_in": "получено от физлица", "external_transfer": "переведено в другой банк",
           "income": "зафиксирован доход"},
}

# what a reason about "nothing happened" checked: shown next to the checked period
PERIOD_WHAT = {
    "uz": {"income": "tushum qayd etilmagan", "cash": "naqd yechish qayd etilmagan", "inflow": "yirik kirim qayd etilmagan",
           "other": "tegishli tranzaksiya topilmagan"},
    "en": {"income": "no income recorded", "cash": "no cash withdrawal recorded", "inflow": "no large inflow recorded",
           "other": "no relevant transaction found"},
    "ru": {"income": "поступлений дохода не зафиксировано", "cash": "снятий наличных не зафиксировано",
           "inflow": "крупных поступлений не зафиксировано", "other": "подходящих транзакций не найдено"},
}


def period_summary(lang: str, start: str, end: str, days: int, group: str) -> str:
    what = PERIOD_WHAT[lang][group]
    if lang == "uz":
        return f"Tekshirilgan davr: {start} dan {end} gacha ({days} kun): {what}"
    if lang == "en":
        return f"Checked period: {start} to {end} ({days} days): {what}"
    return f"Проверенный период: с {start} по {end} ({days} дн.): {what}"


def money(v, lang: str) -> str:
    return f"{int(round(v)):,}".replace(",", " ") + " " + SOM[lang]


def invoice_summary(lang: str, inv: dict) -> str:
    amount, name, oked = money(inv["invoice_amount_uzs"], lang), inv["supplier_name"], inv["supplier_oked"]
    ok = inv["oked_matches_purpose"]
    if lang == "uz":
        return (f"Hisob-faktura {amount}, ta'minotchi {name} (OKED {oked}); faoliyat turi kredit maqsadiga "
                f"{'mos keladi' if ok else 'mos kelmaydi'}")
    if lang == "en":
        return (f"Invoice {amount}, supplier {name} (OKED {oked}); the line of business "
                f"{'matches' if ok else 'does not match'} the loan purpose")
    return (f"Счёт-фактура {amount}, поставщик {name} (ОКЭД {oked}); вид деятельности "
            f"{'соответствует' if ok else 'не соответствует'} цели кредита")


def graph_summary(lang: str, g: dict) -> str:
    cash, span, n = g["cash_share"], g["active_span_days"], g["in_degree"]
    link = bool(g["employee_link"])
    if lang == "uz":
        from .reasons import _d as dd, _pct
        return (f"Ta'minotchi {g['supplier_name']}: {n} ta kreditning hisob-fakturasida ko'rsatilgan; olgan "
                + ("mablag'dan naqd yechmagan; hisobi " if cash < 0.005 else f"mablag'ning {_pct(cash, 'ini')} naqd yechgan; hisobi ")
                + ("hali faol emas; " if span < 0 else "faqat bir kun faol bo'lgan; " if span < 1 else f"{dd(span)} kun faol; ")
                + f"bank xodimi bilan bog'lanish: {'bor' if link else "yo'q"}")
    if lang == "en":
        return (f"Supplier {g['supplier_name']}: appears on the invoices of {n} loans; "
                + ("has not withdrawn cash from the funds received; account " if cash < 0.005
                   else f"withdrew {_p(cash, 'en')} of the funds received in cash; account ")
                + ("not yet active; " if span < 0 else "active for one day only; " if span < 1 else f"active for {_d(span)} days; ")
                + f"link to a bank employee: {'yes' if link else 'no'}")
    return (f"Поставщик {g['supplier_name']}: указан в счетах-фактурах по {n} кредитам; "
            + ("не снимал наличными полученные средства; счёт " if cash < 0.005
               else f"снял наличными {_p(cash, 'ru')} полученных средств; счёт ")
            + ("ещё не активен; " if span < 0 else "был активен всего один день; " if span < 1 else f"активен {_d(span)} дн.; ")
            + f"связь с сотрудником банка: {'есть' if link else 'нет'}")
