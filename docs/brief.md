# MAQSAD — jamoa uchun to'liq loyiha brifi (v1)

> **Claude uchun ko'rsatma (shu faylni o'qiyotgan AI yordamchiga):**
> Bu fayl WIUT Hackathon 2026 finalida qatnashayotgan 3 kishilik jamoaning yagona "manba hujjati".
> Jamoa a'zolari: **Dilshodbek** (AI/ML), **Oybek** (ilova: backend + frontend), **Shohijahon** (biznes, pitch).
> Faylni senga ulardan biri tashlaydi. Avval undan **"Siz kimsiz: Dilshodbek, Oybek yoki Shohijahon?"** deb so'ra
> (agar o'zi aytmagan bo'lsa), keyin faqat o'sha odamning bo'limidagi vazifalarga e'tibor qarat,
> lekin butun loyihani tushunib tur. G'oya **tanlangan va kelishilgan** — uni boshqa g'oyaga almashtirishni taklif qilma;
> vazifang shu g'oyani eng yaxshi qilib qurish va taqdim etishga yordam berish.
> Javoblarni o'zbek tilida, oddiy va aniq yoz. Texnik atamalar inglizcha qolishi mumkin.
> Hech qachon raqam o'ylab topma: hakamlar "invented metrics" uchun diskvalifikatsiya qiladi.

---

## 0. Bir qarashda

| | |
|---|---|
| **Tanlov** | WIUT Hackathon 2026 "Intelligence in Use", **FinTech / AI in Finance** treki, final (10–11 oktyabr 2026) |
| **Partnerlar** | MKBANK va O'zbekiston Markaziy banki |
| **Rejim** | **Partner product** — MKBANK, kredit monitoringi bo'limi (aniq nomini mentorlardan so'raymiz) |
| **Mahsulot nomi** | **Maqsad** |
| **Bir gapda** | Imtiyozli mikrokreditlar haqiqatan aytilgan maqsadga sarflanganini **pul izi, rasm va faktura** dalillari asosida masofadan tekshiradigan AI tizim. Tizim xavf navbatini tuzadi, yakuniy qarorni bank xodimi qiladi. |
| **O'zgaradigan jarayon** | Kreditdan keyingi maqsadli ishlatilish monitoringi: "hammaning oldiga borish" → "xavfga asoslangan, masofadan, qarori inson qo'lida" |
| **Topshiriladigan 3 narsa** | 1) Deploy qilingan ochiq veb-prototip (https link) · 2) Deck (PDF yoki .pptx, ≤ 50 MB) · 3) Ochiq repo: kod, README, datasetlar va litsenziyalar ro'yxati |

---

## 1. Hackaton qoidalari (qisqacha, lekin MUHIM)

**Baholash (har hakam har mezonga 0–10 beradi, vaznli yig'indi):**

| Mezon | Vazn | Hakam nima so'raydi |
|---|---|---|
| Technical complexity | 25% | Bu tizimmi, yoki birovning modelini chaqirishmi? Arxitektura, data pipeline, **o'zimiz qurgan test to'plamida o'lchangan sifat**, bitta so'rov tezligi va narxi, ma'lum xato holatlar |
| Innovation & originality | 25% | **Replication test:** xuddi shu LLM vositalari bor boshqa jamoa buni bir dam olish kunida qayta qura oladimi? Ustunligimiz: data, model, pipeline, integratsiya yoki domen bilimi |
| Impact & practicality | 25% | Muammo realmi, foydalanuvchi kim, kim to'laydi, dalil bormi? 3 oyda regulyatsiya doirasida pilot qilsa bo'ladimi? |
| Presentation | 15% | Aniq hikoya, jonli demo + deploy qilingan sayt, halol raqamlar, to'g'ridan-to'g'ri javoblar, vaqtga rioya |
| Business potential | 10% | Pilot qurish va yuritish xarajati (odamlar, infra, data, litsenziyalar), unit economics, payback muddati, **har bir taxmin yozilgan** |

Durang bo'lsa Technical + Innovation yig'indisi hal qiladi.

**Ruxsat etilgan:** istalgan LLM, coding assistant, open-source model, public dataset/API (pullik xizmatlar cost modelda ko'rsatiladi);
o'zimiz yaratgan ma'lumot, shu jumladan **sintetik data — manbasi va generatsiya usuli e'lon qilinadi**;
saralash bosqichidagi o'z kodimiz (README'da e'lon qilinadi).

**Taqiqlangan (diskvalifikatsiya):**
- Soxta demo: hard-coded javoblar, ishlayotgandek ko'rsatilgan mock ekranlar, **o'ylab topilgan metrikalar**
- Real shaxsiy yoki maxfiy ma'lumot (mijoz yozuvlari, bankning ochiq bo'lmagan ma'lumoti, rozilik bermagan odamlarning yuzi/raqami)
- Finaldan oldin qilingan ishni finalda qurilgandek ko'rsatish
- Atributsiyasiz birovning kodi, modeli, g'oyasi
- "UI orqasida bitta prompt" taqiqlanmagan, lekin ikkala texnik mezonda past ball oladi

**Engineering constraint:** mijoz ma'lumoti bank perimetridan chiqmaydi deb faraz qilinadi. Prototip tashqi API chaqirishi mumkin,
lekin **production dizayni tashqi API'siz ishlashi kerak**. Ko'rsatish kerak: qanday deploy qilinadi, har bir qaror qanday tushuntiriladi va audit qilinadi, inson qayerda nazoratda qoladi.

**Demo:** hakamlar (va boshqa ishtirokchilar) saytni o'z noutbuki va telefonida sinaydi. Prototip **hakam bergan input** bilan uchidan-uchigacha ishlashi shart.

**Deploy va code freeze:** topshirilgan repodan deploy (frontend + backend). GitHub Actions tavsiya etiladi (deploy tarixi saqlanadi).
Deadline'dan keyin faqat topshirilgan commitni qayta deploy qilish mumkin; tashkilotchilar saytlarni kuzatadi.

**Himoya:** 7 daqiqa taqdimot (qat'iy, gap o'rtasida to'xtatiladi) + 3 daqiqa Q&A. Hakam istalgan a'zodan, shu jumladan LLM yozgan kod haqida so'raydi.
Tavsiya etilgan taqsimot: muammo va foydalanuvchi 1 · jonli demo 2 · texnologiya 2 · budjet va ROI 1.5 · keyingi qadamlar 0.5.
Texnik muammo soatni to'xtatmaydi — **demo zaxira videosi shart**.

---

## 2. Muammo (dalillar bilan)

Davlat oilaviy tadbirkorlik va boshqa dasturlar orqali imtiyozli kredit beradi (parranda, chorva, asalarichilik, baliqchilik, issiqxona, tikuv dastgohi va h.k. — ya'ni **jismoniy aktivlar**).
Kreditlarning bir qismi aytilgan maqsadga ishlatilmaydi: aktiv naqdga sotiladi, soxta faktura ko'rsatiladi, pul naqdlashtiriladi.
**Maqsadli ishlatilishni monitoring qilish tijorat banklari zimmasida.** Hozir buni xodim joyiga borib tekshiradi; mijozlar ko'p va tarqoq, shuning uchun hammasini sifatli tekshirib bo'lmaydi.

Dalillar (Oybekning tadqiqotidan; deckka qo'yishdan oldin har birini manbada qayta tekshiring):
- Hududiy tekshiruvlarda **2 500 holatda 68 mlrd so'm** imtiyozli kredit maqsadsiz sarflangani aniqlangan — Prezident yig'ilishida qayd etilgan ([anhor.uz](https://anhor.uz/ru/news/heads-of-banks))
- Oilaviy tadbirkorlik dasturlarida muammoli kreditlar 19% dan 16% ga tushgan, lekin **~3 trln so'm** hali qaytarilmagan (anhor.uz, o'sha manba)
- 16 trln so'mlik loyihalarning 50% i, joriy yildagi 11.5 trln so'mlik loyihalarning 80% i bo'yicha **tushum bankdan tashqarida aylanmoqda**; davlat dasturlari kreditlari ustidan **kunlik nazorat** topshirilgan ([yuz.uz](https://yuz.uz/uz/news/davlat-dasturlari-doirasida-olinayotgan-kreditlarning-maqsadli-ishlatilishi-boyicha-kunlik-nazorat-ornatiladi))
- PQ-5041: monitoring mas'uliyati banklarda; MKBANK kapitaliga shu dasturlar uchun 400 mlrd so'm yo'naltirilgan ([spot.uz](https://www.spot.uz/oz/2024/04/29/credit-money/))
- Kredit shartlari va yo'nalishlari ([gov.uz](https://gov.uz/en/advice/NaN/document/3240)); kreditlar naqd ham beriladi
- MKBANK ko'lami: 162 filial, ~250 ming biznes mijoz, kreditlarning ~82% i mikroqarzlar (Erste Asset Management ma'lumoti)

**Real sxemalar (bizning tipologiyalarimiz shulardan olinadi):**
1. Uskuna olib kelinmasdan, soxta hujjatlar bilan "ishlab chiqarish boshlandi" deyilgan — 8 mlrd so'm ([yuz.uz](https://yuz.uz/uz/news/10-mlrddan-ziyod-imtiyozli-kredit-mablaglari-talon-toroj-qilindi))
2. Kreditga olingan 160 bosh naslli chorva naqdga sotib yuborilgan (spot.uz)
3. Bank xodimi o'z yaqinining xo'jaligini ta'minotchi qilib, 71 fuqaroning 2.18 mlrd so'mlik kreditini naqdlashtirgan ([yuz.uz](https://yuz.uz/uz/news/bank-xodimining-kirdikorlari-fosh-etildi))
4. MKBANK Samarqand filiali murojaati bilan soxta ish haqi ma'lumotlari orqali 386.7 mln so'm o'zlashtirilgani ochilgan; tikuv dastgohlari naqdga sotilgan holat ham bor ([yuz.uz](https://yuz.uz/uz/news/mikroqarz-va-imtiyozli-kredit-mablaglarini-ozlashtirgan-shaxslar-aniqlandi))

---

## 3. Yechim: nima quryapmiz (oddiy tilda)

Bank xodimiga **"kimning oldiga birinchi borish kerak?"** degan savolga javob beradigan tizim. Har bir kredit bo'yicha uch xil dalil yig'iladi:

1. **Pul izi (asosiy, tizimning "miyasi").** Kredit tushgandan keyin pul qayerga ketdi? Kelgan kuni naqd yechildimi? Ta'minotchiga to'lov bo'ldimi?
   Ko'p qarz oluvchining puli bitta "ta'minotchi"ga borib, u yerdan naqdlashtirildimi? Pul bankdan tashqariga chiqib ketdimi?
   Bu bankning o'z ma'lumoti — qarz oluvchi uni soxtalashtira olmaydi.
2. **Rasm.** Qarz oluvchi sotib olgan **aktivni** (odamni emas) sahifadagi jonli kamerada suratga oladi. Tizim tekshiradi:
   a) rasmdagi narsa kredit maqsadiga mosmi (tikuv dastgohi krediti → rasmda dastgoh bormi);
   b) bu rasm portfeldagi boshqa kreditda ishlatilmaganmi (takror rasm);
   c) joylashuv va vaqt (brauzer geolokatsiyasi + server vaqti). Yuz tushib qolsa — xiralashtiriladi, asl rasm saqlanmaydi.
3. **Faktura.** Ta'minotchining faoliyat turi (OKED) kredit maqsadiga mosmi.

Uchala dalil bitta **xavf bahosiga** birlashadi → 🟢 / 🟡 / 🔴. Xodim har kuni tayyor navbatni ko'radi: qaysi kreditlar qizil va **nega**.
LLM tekshiruv dalolatnomasining qoralamasini yozadi (faqat dalillardagi faktlardan, har bir fakt dalilga qarab tekshiriladi).
**Qarorni har doim xodim qiladi** (tasdiqlash / rad etish), har bir qadam audit jurnaliga yoziladi.

**Tizim hech kimni "aybdor" demaydi.** U faqat "bu kreditni birinchi navbatda tekshiring, sabablari mana bular" deydi.

**Nega rasm yolg'iz o'zi yetmaydi:** qo'shnining sigirini suratga olish yoki dastgohni suratga olib keyin sotish mumkin. Shuning uchun asosiy dalil — pul izi. Rasm yaxshi bo'lsa-yu, pul darhol naqdlashtirilgan bo'lsa, kredit baribir qizil bo'ladi.

---

## 4. Arxitektura

```
Qarz oluvchi (telefon)            Bank tizimlari (sintetik portfel)
  rasm + geolokatsiya + faktura      kreditlar + tranzaksiyalar
            │                                  │
            ▼                                  ▼
   ┌──────────────── BANK PERIMETRI (productionda to'liq on-prem) ────────────────┐
   │  Rasm tahlili            Pul izi tahlili             Faktura tahlili          │
   │  - takror (pHash/emb.)   - graf belgilari            - maydonlarni ajratish    │
   │  - aktiv turi (CLIP+probe)- LightGBM + SHAP           - OKED ↔ maqsad mosligi  │
   │            └──────────────┬──────────────┘                     │              │
   │                           ▼                                    │              │
   │                 Xavf bahosi + sabablar  ◄──────────────────────┘              │
   │                           ▼                                                   │
   │          LLM: dalolatnoma qoralamasi (faktlar dalilga qarab tekshiriladi)      │
   │                           ▼                                                   │
   │          Xodim paneli: navbat → kredit sahifasi → TASDIQLASH / RAD ETISH       │
   │                           ▼                                                   │
   │          Audit jurnali (kim, qachon, AI nima tavsiya qildi, xodim nima qildi)  │
   └───────────────────────────────────────────────────────────────────────────────┘
```

- **Prototip:** LLM qismi (va ixtiyoriy faktura ajratish) tashqi API orqali; qolgan hamma narsa (modellar, indeks, navbat) lokal.
- **Production:** lokal LLM (masalan Qwen/Llama, vLLM yoki Ollama orqali) bank serverida; ma'lumot tashqariga chiqmaydi. Buni deckda bitta diagramma bilan ko'rsatamiz.
- **RAG (kichik, ixtiyoriy):** kredit dasturi qoidalari (PQ-5041, summa limitlari, ruxsat etilgan yo'nalishlar) bo'yicha — dalolatnoma "dasturning X-bandiga zid" deb iqtibos keltirishi uchun. Ikkinchi darajali.
- **Agent shart emas:** jarayon aniq pipeline (dalil → baho → dalolatnoma → inson). Hamma narsani "agent" qilish ballni oshirmaydi, demoni beqaror qiladi.

**Muhim:** texnik ball buzzword'lardan (RAG, agent) emas, **o'zimiz qurgan model va o'lchovlardan** keladi.

---

## 5. Repo va hamkorlik (GitHub)

```
maqsad/
├── ml/              ← Dilshodbek: generator, pul izi modeli, rasm tahlili, baholash
├── backend/         ← Oybek: API, LLM dalolatnoma, audit log
├── frontend/        ← Oybek: xodim paneli, qarz oluvchi sahifasi
├── docs/            ← Shohijahon: ROI, manbalar, deck materiallari, Q&A
├── data/README.md   ← barcha datasetlar, rasmlar, modellar va litsenziyalari (TALAB!)
└── README.md        ← ishga tushirish yo'riqnomasi
```

Qoidalar:
- `main` doim ishlaydigan holatda. Har kim o'z branch'ida (`ml/...`, `app/...`, `docs/...`), PR orqali qo'shadi.
- Har kim faqat o'z papkasida ishlaydi → conflict yo'q.
- Katta fayllar (dataset, rasm to'plamlari) repoga qo'yilmaydi (`.gitignore`); model fayli kichik bo'lsa qo'yiladi.
- Tez-tez commit — hakam "finalda yozilganmi?" deb so'rasa, commit tarixi isbot.
- Deploy GitHub Actions orqali. Birinchi soatdayoq deploy qilinsin.

### API shartnomasi (Dilshodbek ↔ Oybek; birinchi 30 daqiqada kelishiladi va o'zgarsa ikkalasi rozi bo'ladi)

```
GET  /queue
  → [{loan_id, borrower_name_masked, purpose, amount_uzs, region, risk, color, top_reason}]
     risk bo'yicha kamayish tartibida

GET  /loan/{loan_id}
  → {loan, transactions[], photos[], invoice?,
     risk, color, reasons:[{code, text_uz, weight}], evidence:[...], model_version}

POST /score_loan        {loan_id}
  → {risk: 0..1, color: "green|yellow|red", reasons[], model_version, latency_ms}

POST /check_photo       {loan_id, image(base64 yoki multipart), lat?, lon?, captured_at}
  → {asset_match: {label, score}, duplicate_of: [{loan_id, similarity}],
     geo_ok, time_ok, faces_blurred: bool, latency_ms}

POST /draft_report      {loan_id}
  → {text_uz, facts:[{claim, evidence_ref}], model, latency_ms, cost_usd}

POST /decision          {loan_id, officer_id, action: "approve|reject|visit", comment}
  → {audit_id}

GET  /audit?loan_id=...
  → [{ts, actor, action, ai_recommendation, model_version, inputs_hash}]
```

Ishlab chiqish paytida Oybek vaqtincha soxta javob qaytaradigan endpoint bilan ishlaydi; **demoda faqat haqiqiy model ishlaydi**.

---

## 6. VAZIFALAR — har bir odam uchun

### 6.1 Dilshodbek — AI/ML qatlami (Technical + Innovation = 50% ball shu yerda)

Dilshodbek ilgari saralash bosqichida AML alert escalation modeli qurgan (LightGBM + LR blend, 147 belgi, halol CV ROC-AUC 0.657).
Tajribasi: pandas, LightGBM, feature engineering, halol CV. U Claude Code subagentlari bilan orkestrator sifatida ishlaydi.

Ustuvorlik tartibida:

**1. Sintetik portfel generatori (ENG BIRINCHI, ~2 soat) — `ml/generator/`**
- 2–5 ming kredit: `loan_id, borrower_id, purpose (5 tur: tikuv_dastgohi, chorva, issiqxona, parranda, asalarichilik), amount_uzs, disbursed_at, region, branch_id, supplier_id`
- Har bir kredit uchun tranzaksiyalar: kredit tushishi, naqd yechish, ta'minotchiga to'lov, P2P o'tkazmalar, keyingi kirimlar
- Ta'minotchilar jadvali: `supplier_id, oked_code, region, linked_employee_id?`
- 4 ta firibgarlik tipologiyasi (real holatlardan):
  - **T1 tez naqdlashtirish:** kredit tushgan kuni/ertasi katta ulushi naqd yechiladi, ta'minotchiga to'lov yo'q
  - **T2 ta'minotchi halqasi:** ko'p qarz oluvchi → bitta ta'minotchi → tez naqdlashtirish; ta'minotchi bank xodimiga bog'liq bo'lishi mumkin
  - **T3 aktivni sotib yuborish:** ta'minotchiga to'lov bor, lekin keyin aktiv sotilganiga o'xshash bir martalik kirim va naqdlashtirish
  - **T4 soxta faktura:** ta'minotchi OKED'i kredit maqsadiga mos emas
- **Halol, lekin shubhali ko'rinadigan** mijozlar (masalan, chorvani qonuniy sotgan fermer, naqd ishlatadigan qishloq xo'jaligi) — model oson "aldanmasligi" uchun
- Test to'plami **boshqa seed va biroz boshqa parametrlar** bilan alohida yaratiladi
- Generatsiya usuli `data/README.md` da to'liq yoziladi (bu qoida talabi)
- Rasmlar uchun: har kreditga ochiq litsenziyali rasmlardan biri bog'lanadi; bir qismi ataylab takrorlanadi (T-rasm sxemasi)

**2. Pul izi modeli (~3 soat) — `ml/money_trail/`**
- Belgilar: kredit → birinchi naqd yechish vaqti; 7/30 kunda naqd ulushi; ta'minotchiga to'lov bormi va ulushi; ta'minotchining in-degree (nechta qarz oluvchi ulangan); ta'minotchidan keyingi naqd oqimi; pul bankdan chiqib ketgan ulush; P2P o'tkazmalar soni; ta'minotchi–xodim bog'liqligi; OKED ↔ maqsad mosligi
- LightGBM + SHAP; har bir SHAP sababi `{code, text_uz}` ko'rinishida oddiy o'zbekchaga o'giriladi
- **Asosiy metrika:** "20% tashrif byudjetida ushlash" (recall@20%) — tasodifiy tanlov va oddiy qoidalar bilan solishtirilgan holda. Qo'shimcha: ROC-AUC, PR-AUC, har bir tipologiya bo'yicha alohida recall
- `score(loan_id) -> dict` funksiyasi API shartnomasiga mos

**3. Rasm takrori (~1.5 soat) — `ml/photo/`**
- Perceptual hash (`imagehash`) va/yoki embedding (masalan CLIP) bo'yicha portfel indeksi; o'xshashlik chegarasi
- Chidamlilik testi: qirqish, siqish (JPEG), o'lcham o'zgarishi, skrinshot — har biri bo'yicha precision/recall
- Yuzni aniqlash va xiralashtirish (oddiy detektor yetarli)

**4. Aktiv turi (keyin) — `ml/photo/`**
- CLIP zero-shot → imkon bo'lsa embedding + o'zimiz o'qitgan linear probe (50–100 ta ochiq litsenziyali rasmda)
- Aniqlik **real rasmlarda** o'lchanadi; litsenziyalar `data/README.md` ga yoziladi

**5. Raqamlar va xato holatlar (deck uchun)**
- Har bir komponent: sifat, p50/p95 kechikish, bitta tekshiruv narxi
- "Known failure cases" ro'yxati (masalan: bitta sigirni farqlay olmaslik, yomon yorug'lik, halol naqd foydalanuvchilar)
- Hammasi `docs/metrics.md` ga, Shohijahon deckka oladi

**Vaqt chegarasi:** bugun kechgacha generator + pul izi + rasm takrori Oybekning paneliga ulangan bo'lishi kerak. Ulgurmasa, aktiv turi va faktura qoldiriladi.

### 6.2 Oybek — ilova (backend + frontend + LLM + audit)

Oybek o'z Claude Code obunasi va subagentlari bilan ishlaydi. Uning qismi — mahsulotning "yuzi": hakamlar ko'radigan va bosib sinaydigan hamma narsa.

1. **Skelet + birinchi soatda deploy** (Render / Railway / HF Spaces / o'z server; GitHub Actions bilan). API shartnomasi bo'yicha vaqtinchalik javoblar.
2. **Xodim paneli:** xavf navbati (🟢🟡🔴, filtr: hudud, maqsad), kredit sahifasi — sabablar, pul izi (oddiy vaqt chizig'i yoki graf), rasmlar yonma-yon, faktura
3. **Qarz oluvchi sahifasi:** jonli kamera (`getUserMedia`), brauzer geolokatsiyasi, server vaqti, rasm yuklash; telefonda qulay
4. **Demo rejimi:** slayddagi QR kod → hakam "tikuv dastgohi" kreditini tanlab, istalgan narsani suratga oladi → natija sekundlarda. Har tashrif buyuruvchiga **alohida sandbox**, rate limit, fayl hajmi cheklovi, admin uchun reset
5. **LLM dalolatnoma qoralamasi:** faqat dalillardagi faktlar; har bir gap `evidence_ref` bilan; narx va kechikish log qilinadi
6. **Tasdiqlash / rad etish / tashrif belgilash** tugmalari va **audit jurnali** (ixtiyoriy: hash-zanjir)
7. **Faktura + OKED (ulgursa):** LLM fakturadan maydonlarni ajratadi (lotin va kirill), OKED ↔ maqsad mosligi
8. **Xatolarga chidamlilik:** noto'g'ri fayl, bo'sh input, juda katta rasm — tizim yiqilmaydi, tushunarli xabar beradi
9. Deck uchun bitta **"deployment & audit"** diagrammasi (on-prem lokal LLM, inson qayerda)

### 6.3 Shohijahon — biznes, dalil, taqdimot (Impact + Presentation + Business = 50%)

Shohijahon biznes talabasi; kod yozmaydi, Claude'dan matn, hisob-kitob va slayd uchun foydalanadi.

1. **DARHOL — mentorlardan so'rash (MKBANK):**
   - Bir yilda nechta maqsadli ishlatilish monitoringi tashrifi qilinadi?
   - Bitta tashrif o'rtacha qancha turadi (vaqt, transport, xodim soati)?
   - Monitoring bo'limining aniq nomi va unda nechta xodim ishlaydi?
   - Ixtiyoriy: hozir tekshiruv qanday hujjatlashtiriladi (dalolatnoma shakli)?
   - Bundan tashqari tashkilotchilardan: deadline aniq vaqti va topshirish tartibi
2. **Dalillarni tekshirish:** 2-bo'limdagi har bir raqamni manbada ochib tekshirish, slayd uchun qisqa formatga keltirish (manba ostida kichik harflar bilan)
3. **ROI modeli (jadval):** tejalgan tashriflar × bitta tashrif narxi + erta ushlangan yo'qotishlar; pilot xarajati (odamlar, GPU server / on-prem, LLM, litsenziyalar); unit economics (bitta tekshiruv narxi); payback muddati. **Har bir taxmin alohida qatorda "Taxmin:" deb yoziladi.** Model raqamlari Dilshodbekdan (`docs/metrics.md`)
4. **3 oylik pilot rejasi:** 1-oy — 1 ta filialda o'rnatish, tarixiy kreditlarda sinash; 2-oy — xodimlar tizim bilan parallel ishlaydi; 3-oy — natijani o'lchash (ushlangan holatlar, tejalgan tashriflar)
5. **Raqobat slaydi:** nega hozirgi usul (hammaga tashrif) va umumiy yechimlar ishlamaydi; bizning farq
6. **Deck + 7 daqiqalik ssenariy:**
   - 1-slayd: "Partner product: MKBANK, kredit monitoringi bo'limi" (talab!)
   - Muammo va foydalanuvchi (1 daq.) → jonli demo (2) → texnologiya (2) → budjet va ROI (1.5) → keyingi qadamlar (0.5)
7. **Q&A tayyorgarligi** (7-bo'lim) va kim qaysi savolga javob berishini taqsimlash
8. **Demo zaxira videosi** (ekran yozuvi), taymer bilan 2–3 marta to'liq mashq

---

## 7. Q&A — tayyor javoblar

| Savol | Javob |
|---|---|
| "Rasmni aldash oson-ku, qo'shnining sigirini suratga oladi" | Rasm yolg'iz qaror emas. Asosiy dalil — pul izi (bankning o'z ma'lumoti). Rasm jonli kamerada, geolokatsiya va server vaqti bilan olinadi; portfel bo'ylab takror qidiriladi. Qizil kreditga baribir xodim boradi. |
| "Model sintetik datada — o'z generatoringizni qayta o'rganyapti-ku" | Test to'plami boshqa seed va parametrlar bilan alohida yaratilgan; halol-shubhali mijozlar qo'shilgan; oddiy qoidalar va tasodifiy tanlov bilan solishtirilgan. Real aniqlik pilotda tarixiy kreditlarda o'lchanadi — buni ochiq aytamiz. |
| "Shaxsiy ma'lumot, biometrika?" | Aktiv suratga olinadi, odam emas; yuz avtomatik xiralashtiriladi, asl rasm saqlanmaydi. Production to'liq on-prem, ma'lumot O'zbekistondagi bank serverida. |
| "Tashqi API'siz ishlaydimi?" | Ha: modellar va indeks lokal; LLM productionda lokal model (vLLM/Ollama). Prototipda tashqi API faqat qulaylik uchun. |
| "Boshqa jamoa buni bir dam olish kunida qura oladimi?" | Rasm yuklash ilovasini — ha. Lekin real prokuratura holatlaridan kodlashtirilgan tipologiyalar, portfel bo'ylab pul izi grafi va rasm indeksi, o'lchangan pipeline va monitoring jarayoniga integratsiyani — yo'q. |
| "Kim to'laydi, qancha tejaydi?" | MKBANK: tejalgan tashriflar va erta ushlangan yo'qotishlar. Raqamlar ROI slaydida, har bir taxmin yozilgan. |
| "Qaror kim tomonidan qabul qilinadi?" | Har doim xodim. Tizim faqat navbat va dalillarni tayyorlaydi; har bir qaror audit jurnalida. |
| "Sigirlarni individual farqlay olasizmi?" | Hozir yo'q — ochiq aytamiz. Yo'l xaritasida: quloq birkasini o'qish. |

---

## 8. Halol xavflar

| Xavf | Yechim |
|---|---|
| Komponentlar ko'p, vaqt kam | MVP = pul izi + rasm takrori + xavf navbati + xodim paneli. Aktiv turi — keyin, faktura — ulgursa |
| Sintetik data tanqidi | 7-bo'limdagi javob; generatsiya usuli ochiq e'lon qilinadi |
| CLIP real rasmlarda past aniqlik | Linear probe; aniqlik o'lchanib, halol ko'rsatiladi |
| Telegram/WhatsApp EXIF'ni o'chiradi | Asosiy yo'l — sahifadagi jonli kamera + geolokatsiya + server vaqti |
| Ochiq demoni boshqalar "ifloslantirishi" | Sandbox, rate limit, reset |
| Demo jonli paytda buziladi | Erta deploy, zaxira video, chidamli xato boshqaruvi |

---

## 9. Vaqt jadvali va nazorat nuqtalari

| Qachon | Nima tayyor bo'lishi kerak |
|---|---|
| **Boshida, 30 daq.** | Repo, papkalar, API shartnomasi, `.gitignore`. Shohijahon mentorlarga savollarni yuboradi |
| **1-kun kechgacha** | Generator + pul izi modeli + rasm takrori → Oybekning paneliga ulangan va **deploy qilingan** (uchidan-uchigacha zanjir) |
| **Kechasi** | Aktiv turi (CLIP), LLM dalolatnoma, audit jurnali, baholash raqamlari (`docs/metrics.md`) |
| **2-kun ertalab** | Faktura (ulgursa), deckka raqamlar, ROI, Q&A javoblari |
| **Deadline'dan 3 soat oldin** | **Yangi funksiya qo'shilmaydi.** Faqat tuzatish, demo videosi, 2–3 marta mashq, topshirish |

Har 3–4 soatda 10 daqiqalik sinxronlash: kim nima qildi, nimaga tiqilib qoldi, API o'zgarishi bormi.

**Oltin qoida:** qayerda to'xtasak ham, **to'liq ishlaydigan** narsa bo'lishi kerak. Chala 5 ta funksiyadan ishlaydigan 3 tasi yaxshi.

---

## 10. Qisqacha eslatmalar (Claude uchun)

- G'oya o'zgarmaydi. Yaxshilash takliflari faqat **shu g'oya doirasida**.
- Raqam o'ylab topma. Noma'lum raqam → "taxmin" deb belgila yoki "o'lchash kerak" de.
- Kod yozganda: API shartnomasiga (5-bo'lim) va papka tuzilishiga rioya qil.
- Har bir dataset, rasm, model, kutubxona litsenziyasi `data/README.md` ga yoziladi.
- Real shaxsiy yoki bank ma'lumoti ishlatilmaydi — faqat sintetik yoki ochiq litsenziyali.
- Demo hech qachon hard-coded javob bermasin.
