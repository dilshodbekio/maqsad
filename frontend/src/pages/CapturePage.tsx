import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type CheckPhotoResult, type Color, type LoanDetail, type QueueItem } from "../api";
import { COLOR_ICON, COLOR_UZ, formatRisk, formatUzs, purposeName } from "../format";
import { onLinkClick } from "../router";
import { BlockError, BlockLoading } from "./loan/evidence";
import "./capture.css";

const MAX_SIDE = 1600;
const JPEG_QUALITY = 0.85;
const GEO_TIMEOUT_MS = 8000;
const PICKER_LIMIT = 200;

type Coords = { lat: number; lon: number } | null;
type Outcome = { result: CheckPhotoResult; before: LoanDetail | null; after: LoanDetail | null };

const loanHref = (id: string) => `/loan/${encodeURIComponent(id)}`;

function getCoords(): Promise<Coords> {
  return new Promise((resolve) => {
    if (!("geolocation" in navigator)) return resolve(null);
    navigator.geolocation.getCurrentPosition(
      (p) => resolve({ lat: p.coords.latitude, lon: p.coords.longitude }),
      () => resolve(null), // denied / unavailable: the check goes on without a location
      { enableHighAccuracy: true, timeout: GEO_TIMEOUT_MS, maximumAge: 60000 },
    );
  });
}

/** Draw the image on a canvas (max side 1600 px) and encode it as JPEG; EXIF orientation is applied by the browser. */
async function compress(source: Blob | HTMLVideoElement): Promise<Blob> {
  let bitmap: ImageBitmap | null = null;
  let w: number;
  let h: number;
  if (source instanceof HTMLVideoElement) {
    w = source.videoWidth;
    h = source.videoHeight;
  } else {
    bitmap = await createImageBitmap(source, { imageOrientation: "from-image" });
    w = bitmap.width;
    h = bitmap.height;
  }
  if (!w || !h) throw new Error("empty frame");
  const k = Math.min(1, MAX_SIDE / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * k);
  canvas.height = Math.round(h * k);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no canvas");
  ctx.drawImage(source instanceof HTMLVideoElement ? source : (bitmap as ImageBitmap), 0, 0, canvas.width, canvas.height);
  bitmap?.close();
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob"))), "image/jpeg", JPEG_QUALITY),
  );
}

export function CapturePage({ search, ready }: { search: string; ready: boolean }) {
  const initialLoan = useMemo(() => new URLSearchParams(search).get("loan") ?? "", [search]);
  const [loanId, setLoanId] = useState(initialLoan);
  const [manualId, setManualId] = useState("");
  const [queue, setQueue] = useState<QueueItem[] | null>(null);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [loan, setLoan] = useState<LoanDetail | null>(null);
  const [loanError, setLoanError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const geoRef = useRef<Promise<Coords> | null>(null);
  const [camera, setCamera] = useState<"off" | "starting" | "on" | "unavailable">("off");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => setLoanId(initialLoan), [initialLoan]);

  // loan picker: the queue of this sandbox (top PICKER_LIMIT items)
  useEffect(() => {
    if (!ready || initialLoan) return;
    api
      .queue({}, 0, PICKER_LIMIT)
      .then((q) => setQueue(q.items))
      .catch((e: Error) => setQueueError(e.message));
  }, [ready, initialLoan]);

  // the selected loan: purpose and current colour
  const loadLoan = useCallback(() => {
    if (!ready || !loanId) {
      setLoan(null);
      return;
    }
    setLoanError(null);
    api
      .loan(loanId)
      .then(setLoan)
      .catch((e: Error) => {
        setLoan(null);
        setLoanError(e.message);
      });
  }, [ready, loanId]);
  useEffect(loadLoan, [loadLoan]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCamera("off");
  }, []);
  useEffect(() => stopCamera, [stopCamera]);
  useEffect(
    () => () => {
      if (preview) URL.revokeObjectURL(preview);
    },
    [preview],
  );

  const startCamera = useCallback(async () => {
    setError(null);
    geoRef.current = getCoords(); // ask for the location together with the camera
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamera("unavailable");
      return;
    }
    setCamera("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1440 } },
        audio: false,
      });
      streamRef.current = stream;
      setCamera("on");
    } catch {
      setCamera("unavailable");
    }
  }, []);

  useEffect(() => {
    const v = videoRef.current;
    if (camera === "on" && v && streamRef.current) {
      v.srcObject = streamRef.current;
      v.play().catch(() => undefined);
    }
  }, [camera]);

  const send = useCallback(
    async (blob: Blob, name: string) => {
      if (!loanId) return;
      setBusy(true);
      setError(null);
      setOutcome(null);
      try {
        const coords = await (geoRef.current ?? getCoords());
        const form = new FormData();
        form.append("loan_id", loanId);
        form.append("image", blob, name);
        if (coords) {
          form.append("lat", String(coords.lat));
          form.append("lon", String(coords.lon));
        }
        form.append("captured_at", new Date().toISOString());
        const before = loan;
        const result = await api.checkPhoto(form);
        const after = await api.loan(loanId).catch(() => null);
        setOutcome({ result, before, after });
        if (after) setLoan(after);
        stopCamera();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Rasmni tekshirib bo'lmadi. Qayta urinib ko'ring.");
      } finally {
        setBusy(false);
      }
    },
    [loanId, loan, stopCamera],
  );

  const shoot = useCallback(async () => {
    const v = videoRef.current;
    if (!v) return;
    try {
      const blob = await compress(v);
      setPreview(URL.createObjectURL(blob));
      await send(blob, "photo.jpg");
    } catch {
      setError("Suratga olib bo'lmadi. Qayta urinib ko'ring yoki rasmni fayldan yuklang.");
    }
  }, [send]);

  const onFile = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      geoRef.current ??= getCoords();
      let blob: Blob = file;
      try {
        blob = await compress(file);
      } catch {
        // the browser cannot decode it: send the original, the server validates the type
      }
      setPreview(URL.createObjectURL(blob));
      await send(blob, blob === file ? file.name : "photo.jpg");
    },
    [send],
  );

  const reset = () => {
    setOutcome(null);
    setError(null);
    setPreview(null);
  };

  const canShoot = Boolean(loanId) && !busy && ready && !loanError;

  return (
    <section className="card capture">
      <h2>Rasm tekshirish</h2>
      <p className="muted small">
        Kredit hisobidan olingan aktivning (uskuna, chorva, issiqxona va h.k.) suratini oling. Tizim rasmni xodimga
        ko'rsatish uchun tekshiradi; yakuniy qarorni bank xodimi qabul qiladi.
      </p>

      {!ready && <BlockLoading text="Model tayyorlanmoqda… Bir necha soniyadan keyin davom eting." />}

      {ready && !initialLoan && (
        <>
          <label className="capture-field">
            Kredit
            <select
              value={loanId}
              onChange={(e) => {
                setLoanId(e.target.value);
                reset();
              }}
              disabled={!queue}
            >
              <option value="">
                {queue ? "— kreditni tanlang —" : queueError ? "Ro'yxat yuklanmadi" : "Yuklanmoqda…"}
              </option>
              {(queue ?? []).map((q) => (
                <option key={q.loan_id} value={q.loan_id}>
                  {q.loan_id} · {purposeName(q.purpose)} · {q.borrower_name_masked}
                </option>
              ))}
            </select>
          </label>
          {queueError && <p className="error small">{queueError}</p>}
          <label className="capture-field">
            Yoki kredit raqamini kiriting
            <input
              value={manualId}
              placeholder="kredit raqami"
              autoCapitalize="characters"
              onChange={(e) => setManualId(e.target.value.trim())}
              onBlur={() => manualId && setLoanId(manualId)}
              onKeyDown={(e) => e.key === "Enter" && manualId && setLoanId(manualId)}
            />
          </label>
        </>
      )}

      {loanError && <BlockError message={loanError} onRetry={loadLoan} />}
      {loan && (
        <p className="capture-loan">
          Kredit <strong>{loan.loan.loan_id}</strong> · maqsad:{" "}
          <strong>{loan.loan.purpose ? purposeName(loan.loan.purpose) : "—"}</strong>
          {typeof loan.loan.amount_uzs === "number" && <> · {formatUzs(loan.loan.amount_uzs)}</>}
        </p>
      )}

      {!outcome && (
        <>
          {camera === "on" && (
            <div className="cam-box">
              <video ref={videoRef} playsInline muted autoPlay />
            </div>
          )}
          {busy && preview && (
            <div className="cam-box">
              <img src={preview} alt="Olingan rasm" />
            </div>
          )}
          {busy && <BlockLoading text="Rasm tekshirilmoqda…" />}

          <div className="capture-actions">
            {camera === "on" ? (
              <button type="button" className="shutter" onClick={shoot} disabled={!canShoot}>
                📷 Suratga olish
              </button>
            ) : (
              <button
                type="button"
                className="shutter"
                onClick={startCamera}
                disabled={!canShoot || camera === "starting"}
              >
                {camera === "starting" ? "Kamera yoqilmoqda…" : "📷 Kamerani yoqish"}
              </button>
            )}
            {camera === "unavailable" && (
              <p className="error small" role="alert">
                Kameraga ruxsat berilmadi yoki kamera topilmadi. Rasmni fayldan yuklang.
              </p>
            )}
            <label className={`secondary-file${canShoot ? "" : " disabled"}`}>
              <span className="muted small">Yoki rasmni fayldan yuklang</span>
              <input
                type="file"
                accept="image/*"
                disabled={!canShoot}
                onChange={(e) => {
                  void onFile(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </label>
            {!loanId && ready && <p className="muted small">Avval kreditni tanlang.</p>}
          </div>
          {error && <BlockError message={error} />}
        </>
      )}

      {outcome && <CaptureResult outcome={outcome} loanId={loanId} onAgain={reset} />}
    </section>
  );
}

function CaptureResult({ outcome, loanId, onAgain }: { outcome: Outcome; loanId: string; onAgain: () => void }) {
  const { result: r, before, after } = outcome;
  const am = r.asset_match;
  const geo =
    r.geo_ok === null
      ? ["Joylashuv berilmadi", ""]
      : r.geo_ok
        ? ["Joylashuv kredit hududiga mos", "ok"]
        : ["Joylashuv kredit hududiga mos emas", "warn"];
  const time =
    r.time_ok === null
      ? ["Vaqt berilmadi", ""]
      : r.time_ok
        ? ["Surat vaqti mos", "ok"]
        : ["Surat vaqti mos emas", "warn"];
  const changed: [Color, Color] | null =
    before && after && before.color !== after.color ? [before.color, after.color] : null;
  const reasons = (after?.escalations ?? []).filter((e) => e.raises_color);

  return (
    <div className="cap-result" aria-live="polite">
      <h3 className="sub-h">Natija</h3>
      <div className="cap-row">
        <strong>Aktiv turi</strong>
        {am ? (
          <span>
            {am.label_uz} <span className="muted">(baho {formatRisk(am.score)})</span>
          </span>
        ) : (
          <span>Aktiv turi aniqlanmadi</span>
        )}
      </div>
      {am && am.matches_purpose === false && (
        <p className="cap-warning" role="alert">
          Rasmdagi narsa kredit maqsadiga mos emas
        </p>
      )}
      <div className={`cap-row ${r.duplicate_of.length ? "warn" : "ok"}`}>
        <strong>Dublikat</strong>
        {r.duplicate_of.length ? (
          <span>
            Bu rasm boshqa kredit(lar) bilan o'xshash:{" "}
            {r.duplicate_of.map((d, i) => (
              <span key={d.loan_id}>
                {i > 0 && ", "}
                <a href={loanHref(d.loan_id)} onClick={(e) => onLinkClick(e, loanHref(d.loan_id))}>
                  {d.loan_id}
                </a>
              </span>
            ))}
          </span>
        ) : (
          <span>Boshqa kreditlarda o'xshash rasm topilmadi</span>
        )}
      </div>
      <div className={`cap-row ${geo[1]}`}>
        <strong>Joylashuv</strong>
        <span>{geo[0]}</span>
      </div>
      <div className={`cap-row ${time[1]}`}>
        <strong>Vaqt</strong>
        <span>{time[0]}</span>
      </div>
      <div className="cap-row">
        <strong>Maxfiylik</strong>
        <span>
          {r.faces_blurred
            ? `Yuzlar xiralashtirildi${typeof r.faces_count === "number" ? ` (${r.faces_count} ta)` : ""}`
            : "Rasmda yuz topilmadi"}
        </span>
      </div>
      {changed && (
        <div className="cap-row warn">
          <strong>Navbatdagi rang o'zgardi</strong>
          <span className="cap-color">
            {COLOR_ICON[changed[0]]} {COLOR_UZ[changed[0]]} → {COLOR_ICON[changed[1]]} {COLOR_UZ[changed[1]]}
          </span>
          {reasons.map((e) => (
            <span key={e.code}>{e.text_uz}</span>
          ))}
          <span className="muted small">Bu tekshirishda ustuvorlik berish belgisi; xodim yakuniy qaror qiladi.</span>
        </div>
      )}
      <div className="capture-actions">
        <a className="capture-link" href={loanHref(loanId)} onClick={(e) => onLinkClick(e, loanHref(loanId))}>
          Kredit sahifasini ochish
        </a>
        <button type="button" className="secondary" onClick={onAgain}>
          Yana rasm olish
        </button>
      </div>
    </div>
  );
}
