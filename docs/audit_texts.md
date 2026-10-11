# Text audit (user-facing strings in Uzbek, English and Russian)

Produced by `python -m ml.eval.audit_texts --tag before|after`. Every reason, escalation, `top_reason`, evidence summary, scenario reason, `ScenarioError` message and asset label of: all loans of the demo portfolio (default view), the 4 scenario presets, 30 random valid scenarios, the invalid scenarios and `check_photo` on 20 committed images in a fresh sandbox is checked for: None / nan / inf, braces, '0 foiz', 'atigi 0', impossible negative numbers, > 100% shown as percent, English words, duplicated reasons, `top_reason` != `reasons[0]`, label / matches_purpose contradictions, empty or very short strings, double spaces, unbalanced brackets, lowercase start, percent-suffix style.

**How to read the table.** 'before' = this checker run on the templates as they were before this round; 'after' = the final templates. The checker itself was corrected after its first run (it had counted the Uzbek word "to'lanmagan" as English 'to' 3 984 times and treated field names in brackets as English); the numbers below are from the corrected checker for both columns.

| Check | before | after |
|---|---|---|
| 0 foiz / 0 ta / 0.0 kun | 136 | 0 |
| > 100% shown as percent | 12 | 0 |
| **strings checked** | 36103 | 37156 |
| unique strings | 14093 | 14471 |
| translated texts checked (each in uz / en / ru) | not checked | 37156 |

**After the fixes the audit finds nothing to flag.**

**Languages.** Before this round the ML layer produced Uzbek texts only (0 English, 0 Russian). Now each of the 37156 texts checked (reasons, escalations, `top_reason`, evidence summaries, scenario reasons / errors / presets, asset labels) exists in Uzbek, English and Russian; the audit finds in none of them an empty or untranslated string, `None` / `nan` / braces, a zero-value phrase, or a number that differs between the three languages.

## Ten random reasons from the final run

- Hisobga kredit summasining 42 foiziga teng bitta yirik kirim tushgan
- Ta'minotchi hisobi atigi 6.8 kun faol bo'lgan
- Yirik kirimdan keyingi 5 kunda uning 79 foizi hisobdan chiqarilgan
- Eng yirik kirim kredit tushgandan 6.0 kun o'tib kelgan
- Oxirgi tushumdan (yoki tushum kutilgan sanadan) beri 23 kun o'tgan
- Hisob-fakturadagi summaning 92 foizi ta'minotchiga to'lanmagan
- Ta'minotchi hisobi 121 kun faol bo'lgan
- Yirik kirimdan keyingi 5 kunda uning 73 foizi hisobdan chiqarilgan
- Ta'minotchi birinchi to'lovdan 668 kun oldin ro'yxatdan o'tgan
- Ta'minotchi birinchi to'lovdan 952 kun oldin ro'yxatdan o'tgan

## Ten random reasons in all three languages

| uz | en | ru |
|---|---|---|
| Ta'minotchi pulni olgandan 2.1 kun o'tib naqd yechgan | The supplier withdrew cash 2.1 days after receiving the money | Поставщик снял наличные через 2.1 дн. после получения денег |
| Ta'minotchi hisobi atigi 57 kun faol bo'lgan | The supplier's account has been active for only 57 days | Счёт поставщика был активен лишь 57 дн. |
| Ta'minotchining faoliyat turi (OKED 96.09) kredit maqsadiga mos kelmaydi | The supplier's line of business (OKED 96.09) does not match the loan purpose | Вид деятельности поставщика (ОКЭД 96.09) не соответствует цели кредита |
| Ta'minotchining faoliyat turi (OKED 68.20) kredit maqsadiga mos kelmaydi | The supplier's line of business (OKED 68.20) does not match the loan purpose | Вид деятельности поставщика (ОКЭД 68.20) не соответствует цели кредита |
| Hisobga kredit summasining 32 foiziga teng bitta yirik kirim tushgan | One large inflow equal to 32% of the loan amount was received | Поступил один крупный платёж, равный 32% суммы кредита |
| Eng yirik kirim kredit tushgandan 6.7 kun o'tib kelgan | The largest inflow arrived 6.7 days after disbursement | Самое крупное поступление пришло через 6.7 дн. после выдачи кредита |
| Kredit tushganidan beri mablag'ning 72 foizi naqd yechilgan | 72% of the funds have been withdrawn in cash since disbursement | С момента выдачи кредита снято наличными 72% средств |
| Ta'minotchining faoliyat turi (OKED 49.41) kredit maqsadiga mos kelmaydi | The supplier's line of business (OKED 49.41) does not match the loan purpose | Вид деятельности поставщика (ОКЭД 49.41) не соответствует цели кредита |
| Kredit tushganidan beri mablag'ning 43 foizi naqd yechilgan | 43% of the funds have been withdrawn in cash since disbursement | С момента выдачи кредита снято наличными 43% средств |
| Ta'minotchi birinchi to'lovdan 1631 kun oldin ro'yxatdan o'tgan | The supplier was registered 1631 days before the first payment | Поставщик зарегистрирован за 1631 дн. до первого платежа |

Unique texts per language in the run: uz 14471, en 14471, ru 14471.

