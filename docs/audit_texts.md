# Text audit (user-facing Uzbek strings)

Produced by `python -m ml.eval.audit_texts --tag before|after`. Every reason, escalation, `top_reason`, evidence summary, scenario reason, `ScenarioError` message and asset label of: all loans of the demo portfolio (default view), the 4 scenario presets, 30 random valid scenarios, the invalid scenarios and `check_photo` on 20 committed images in a fresh sandbox is checked for: None / nan / inf, braces, '0 foiz', 'atigi 0', impossible negative numbers, > 100% shown as percent, English words, duplicated reasons, `top_reason` != `reasons[0]`, label / matches_purpose contradictions, empty or very short strings, double spaces, unbalanced brackets, lowercase start, percent-suffix style.

**How to read the table.** 'before' = this checker run on the templates as they were before this round; 'after' = the final templates. The checker itself was corrected after its first run (it had counted the Uzbek word "to'lanmagan" as English 'to' 3 984 times and treated field names in brackets as English); the numbers below are from the corrected checker for both columns.

| Check | before | after |
|---|---|---|
| 0 foiz / 0 ta / 0.0 kun | 136 | 0 |
| > 100% shown as percent | 12 | 0 |
| **strings checked** | 36103 | 36103 |
| unique strings | 14093 | 14086 |

**After the fixes the audit finds nothing to flag.**

## Ten random reasons from the final run

- Hisobga kredit summasining 42 foiziga teng bitta yirik kirim tushgan
- Ta'minotchi hisobi atigi 6.8 kun faol bo'lgan
- Yirik kirimdan keyingi 5 kunda uning 76 foizi hisobdan chiqarilgan
- Eng yirik kirim kredit tushgandan 6.0 kun o'tib kelgan
- Oxirgi tushumdan (yoki tushum kutilgan sanadan) beri 23 kun o'tgan
- Hisob-fakturadagi summaning 92 foizi ta'minotchiga to'lanmagan
- Ta'minotchi hisobi 121 kun faol bo'lgan
- Yirik kirimdan keyingi 5 kunda uning 70 foizi hisobdan chiqarilgan
- Ta'minotchi birinchi to'lovdan 668 kun oldin ro'yxatdan o'tgan
- Ta'minotchi birinchi to'lovdan 952 kun oldin ro'yxatdan o'tgan

