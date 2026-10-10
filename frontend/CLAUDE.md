# frontend/ — qoidalar

Ildizdagi `CLAUDE.md`, `backend/CLAUDE.md` va `docs/api_contract.md` amal qiladi. Bu yerda faqat frontendga xos qoidalar.

## Stack
- React + Vite (TypeScript). Build natijasi `frontend/dist`; uni FastAPI `/` da beradi. Barcha so'rovlar nisbiy `/api/...` ga.
- Dev'da Vite proxy: `/api` → `http://localhost:8000`.
- Og'ir UI kutubxonalari yo'q; oddiy CSS yoki Tailwind. Interfeys tili — o'zbekcha (lotin).

## Sahifalar
1. `/` — **xodim paneli**: xavf navbati (🟢🟡🔴, risk bo'yicha kamayish tartibida), filtr: hudud, maqsad.
2. `/loan/:id` — **kredit sahifasi**: sabablar (`reasons[].text_uz` + og'irligi), pul izi vaqt chizig'i, rasmlar yonma-yon,
   faktura, LLM dalolatnoma (har fakt yonida dalil havolasi; `unverified` faktlar ajratib ko'rsatiladi),
   tugmalar: Tasdiqlash / Rad etish / Tashrif belgilash, pastda audit jurnali.
3. `/capture` — **qarz oluvchi sahifasi (telefon uchun)**: kredit tanlash → jonli kamera (`getUserMedia`, orqa kamera
   `facingMode: "environment"`), geolokatsiya ruxsati, suratga olish → `/api/check_photo` → natija.
   Kamera ruxsati berilmasa yoki ishlamasa — fayl yuklash zaxira yo'li.

## Majburiy
- `GET /api/health` da `ml_mode == "stub"` bo'lsa, har sahifa tepasida sariq **"STUB — haqiqiy model ulanmagan"** banner.
- Hech qanday natija frontendda qo'lda yozilmaydi; ko'rsatilgan har bir raqam API'dan keladi.
- Har so'rovda loading holati va o'zbekcha xato xabari; tizim hech qachon oq ekran bo'lib qolmaydi.
- Mobil birinchi: 360px kenglikda ishlashi kerak (hakamlar telefonda sinaydi).
- Tizim hech kimni "aybdor" demaydi: matnlarda "xavf", "birinchi navbatda tekshirish" iboralari ishlatiladi.
- Git: commit/push qilma — Oybek o'zi qiladi (batafsil `backend/CLAUDE.md` dagi "Git" bo'limida).
