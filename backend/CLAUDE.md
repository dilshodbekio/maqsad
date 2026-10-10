# backend/ — qoidalar (Oybek va uning subagentlari uchun)

Ildizdagi `CLAUDE.md` va `docs/brief.md` ustun turadi. Bu fayl faqat `backend/` ga xos qoidalarni qo'shadi.
Ish boshlashdan oldin o'qing: `CLAUDE.md`, `docs/api_contract.md`, `docs/brief.md` (4, 5, 6.2-bo'limlar).

## Doira
- Faqat `backend/`, `frontend/`, ildizdagi `Dockerfile`, `railway.json`, `.github/workflows/` da ishlaymiz.
- `ml/` ga **tegmaymiz** (Dilshodbekning papkasi). U yerda o'zgarish kerak bo'lsa — to'xtab, Oybekka ayt.
- Branch: `app/<qisqa-nom>`, `main` ga faqat PR orqali. `main` doim ishlaydigan holatda.

## Stack
- Python 3.13, FastAPI, Uvicorn, Pydantic v2. Saqlash: SQLite (`DATA_DIR` env, default `./data_runtime`).
- Bitta servis: FastAPI API'ni `/api/*` da, frontend build'ini (`frontend/dist`) `/` da static sifatida beradi.
- Port: `PORT` env (default 8000). Deploy: Railway, ildizdagi `Dockerfile` orqali.
- Backend kutubxonalari `backend/requirements.txt` da; ML kutubxonalari `ml/requirements.txt` da (Dockerfile ikkalasini o'rnatadi).

## API shartnomasi
- Yagona manba: `docs/api_contract.md`. Endpoint, maydon nomi yoki turini **o'zboshimchalik bilan o'zgartirma**.
  O'zgarish kerak bo'lsa — kodni yozmasdan oldin Oybekka ayt (Dilshodbek bilan kelishiladi).
- Har bir endpoint uchun Pydantic sxema `backend/app/schemas.py` da, shartnomaga 1:1 mos.

## ML integratsiyasi
- ML faqat bitta joy orqali chaqiriladi: `backend/app/ml_adapter.py`.
  - `from ml.api import score_loan, check_photo, get_queue, get_loan` muvaffaqiyatli bo'lsa → `ML_MODE = "real"`.
  - `ImportError` bo'lsa → `backend/app/ml_stub.py` → `ML_MODE = "stub"`.
- **Stub qoidalari (diskvalifikatsiya xavfi!):**
  - Stub javoblar sintetik namunaviy fayldan (`backend/fixtures/`) o'qiladi, kodda qo'lda yozilgan "chiroyli" javob yo'q.
  - Har bir stub javobda `model_version: "stub"`.
  - `GET /api/health` → `{ml_mode, model_version}`; frontend `ml_mode == "stub"` bo'lsa ekranda **ko'rinadigan "STUB" banner** chiqaradi.
  - Production'da `REQUIRE_REAL_ML=1` bo'lsa va ML import bo'lmasa — servis ishga tushmaydi (jim stubga tushib qolmaydi).
- ML funksiyasidan xato kelsa: 500 emas, tushunarli xabar bilan 502/422, xato log qilinadi.

## Rasm va shaxsiy ma'lumot
- Fayl cheklovi: ≤ 8 MB, faqat JPEG/PNG/WebP; noto'g'ri fayl → 400 va o'zbekcha xabar.
- **Asl rasm diskka yozilmaydi.** Faqat yuzi xiralashtirilgan versiya saqlanadi (xiralashtirishni `ml` qiladi — interfeys Dilshodbek bilan kelishiladi).
- Geolokatsiya ixtiyoriy; `captured_at` uchun **server vaqti** asosiy, mijoz vaqti faqat solishtirish uchun.
- Hech qachon real shaxsiy ma'lumot, real bank ma'lumoti ishlatilmaydi. Ismlar faqat maskalangan holda qaytariladi.

## Demo sandbox
- Har bir tashrif buyuruvchiga `sandbox_id` cookie (uuid). Hakam yuklagan rasmlar va qarorlar faqat o'z sandbox'ida ko'rinadi;
  takror qidiruv: portfel indeksi + o'z sandbox'i.
- Rate limit: IP/sandbox bo'yicha (masalan 20 so'rov/daqiqa `check_photo` va `draft_report` uchun).
- `POST /api/admin/reset` — faqat `ADMIN_TOKEN` header bilan.

## LLM dalolatnoma (`/draft_report`)
- OpenAI-compatible klient, sozlamalar env'dan: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`.
  Prototipda tashqi API; productionda shu kod o'zgarmasdan lokal vLLM/Ollama'ga ulanadi (faqat `LLM_BASE_URL`).
- Promptga faqat shu kreditning dalillari (`evidence[]`, `reasons[]`) beriladi. Javob JSON: `{text_uz, facts:[{claim, evidence_ref}]}`.
- **Fakt tekshiruvi (kodda, LLM'siz):** har bir `evidence_ref` haqiqatan mavjudmi; claim'dagi har bir raqam dalilda bormi.
  O'tmagan fakt olib tashlanadi yoki `unverified: true` bilan belgilanadi — yashirilmaydi.
- Har chaqiruv log qilinadi: `model, latency_ms, input/output tokens, cost_usd`.
- LLM ishlamasa: dalolatnomasiz sahifa ishlayveradi, "qoralama tayyor emas" xabari.

## Qaror va audit
- `/decision`: `approve | reject | visit`; qarorni faqat inson qiladi, AI tavsiyasi alohida maydonda saqlanadi.
- Audit jadvali append-only: `ts, actor, action, ai_recommendation, model_version, inputs_hash, prev_hash, hash`
  (hash-zanjir: `hash = sha256(prev_hash + qator)`). UPDATE/DELETE yo'q.
- Audit va sandbox ma'lumotlari `DATA_DIR` da (Railway volume'ga ulanadi).

## Sifat
- Har endpoint uchun kamida bitta `pytest` (FastAPI TestClient), jumladan yomon input: bo'sh body, katta fayl, noto'g'ri tur, mavjud bo'lmagan `loan_id`.
- Har javobda `latency_ms` (shartnomada bo'lsa). Raqam o'ylab topilmaydi — metrikalar faqat o'lchangan bo'lsa ko'rsatiladi.
- Yangi kutubxona qo'shilsa → `data/README.md` dagi litsenziyalar ro'yxatiga yozish uchun Oybekka ayt.
## Git
- **`git commit`, `git push`, `git checkout`, `git merge`, `git reset` qilma.** Commitlarni Oybek o'zi qo'lda qiladi.
- Har bir mantiqiy bosqich oxirida to'xta va yoz: qaysi fayllar o'zgardi, testlar o'tdimi, tavsiya etilgan commit xabari (inglizcha, qisqa).
- `.env`, API kalitlar, `data_runtime/`, `node_modules/`, `frontend/dist/` hech qachon commitga tushmasin — `.gitignore` da borligini tekshir.
