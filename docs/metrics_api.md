# Ilova (backend) kechikish o'lchovlari

Faqat o'lchangan raqamlar. Qayta ishlab chiqarish: `python backend/scripts/measure_api.py` (ixtiyoriy `--base`, `--n`, `--only`).

## Usul
- Sayt: https://maqsad-production-d704.up.railway.app (model `money_trail_v2`, ml_mode `real`).
- O'lchov sanasi: 2026-10-10 21:12 UTC (JSON'dagi `measured_at_utc`).
- Mijoz: Python `httpx`, ketma-ket so'rovlar, dasturchi noutbukidan. **Tarmoq vaqti kiradi**: bu serverning sof ishlov vaqti emas.
- Har endpoint uchun n = 20 so'rov. Persentil: saralangan namunada eng yaqin o'rin (nearest-rank); n = 20 da p95 = 19-eng sekin so'rov.
- check_photo so'rovlari orasida 3,2 s pauza (ilovaning 20 so'rov/daqiqa chegarasi uchun).

## Natijalar (n = 20, xato = 0)

| Endpoint | p50, ms | p95, ms | min–max, ms |
|---|---|---|---|
| `GET /api/queue?limit=50` | 292,2 | 550,8 | 283,3–569,6 |
| `GET /api/loan/{id}` (50 kredit bo'ylab aylanib) | 232,1 | 241,4 | 223,1–241,6 |
| `POST /api/check_photo` (`ml/assets/photos/asalarichilik_00.jpg`, 38 KB) | 362,9 | 375,3 | 352,7–576,9 |

`check_photo` javobidagi server tomon `latency_ms`: p50 124,1 ms, p95 136,1 ms (model ishlov vaqti; qolgani tarmoq va yuklash).

## Dalolatnoma (`/api/draft_report`)

**O'lchanmagan.** Kesh va `regenerate` bu bosqichda yozilgan, production'ga hali deploy qilinmagan; bitta dalolatnomaning tokenlari va
`cost_usd` qiymatlari ham deploydan keyin o'lchanadi (`--only draft_cached,draft_fresh`).

Kodni lokal sinashda (production emas, har biri 1 ta qoralama, 3 kredit: qizil/sariq/yashil; bu p50/p95 emas):

| `reasoning_effort` | kechikish, s (qizil / sariq / yashil) | tasdiqlanmagan fakt |
|---|---|---|
| yuborilmagan (modelning o'z "thinking"i) | 11,5 / 7,8 / 17,0 | 0 / 0 / 1 |
| `none` | 3,7 / 5,8 / 2,5 | 0 / 0 / 1 |
| `minimal` | 3,6 / (kvota xatosi) / (kvota xatosi) | 0 / – / – |

`minimal` bilan yashil va sariq kreditlar Gemini kvotasi (429) tufayli o'lchanmadi. Sifat: qizil kredit matni dalil raqamlariga mos
(fakt tekshiruvidan 5 dan 5 o'tdi). Xulosa 3 ta namunaga tayanadi, statistik emas.

## Cheklov: Gemini kvotasi
Lokal `.env` dagi kalit bepul tarifda: Google 429 javobida `generate_content_free_tier_requests`, chegara 5 so'rov/daqiqa
(`gemini-3.6-flash`) deb yozdi. Production kaliti haqida bu ma'lumot yo'q. Keshsiz o'lchov 13 s pauza bilan olinadi.
