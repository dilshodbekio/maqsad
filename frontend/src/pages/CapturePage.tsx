import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type CheckPhotoResult, type Color, type LoanDetail, type QueueItem } from "../api";
import { COLOR_ICON, colorName, escalationText, formatRisk, formatUzs, purposeName } from "../format";
import { localized, t } from "../i18n";
import { tn } from "../i18n/jsx";
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
        setError(e instanceof Error ? e.message : t("capture.checkFailed"));
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
      setError(t("capture.shootFailed"));
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
      <h2>{t("capture.title")}</h2>
      <p className="muted small">{t("capture.intro")}</p>

      {!ready && <BlockLoading text={t("capture.warming")} />}

      {ready && !initialLoan && (
        <>
          <label className="capture-field">
            {t("capture.loan")}
            <select
              value={loanId}
              onChange={(e) => {
                setLoanId(e.target.value);
                reset();
              }}
              disabled={!queue}
            >
              <option value="">
                {queue ? t("capture.choose") : queueError ? t("capture.listError") : t("common.loading")}
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
            {t("capture.orEnter")}
            <input
              value={manualId}
              placeholder={t("capture.idPlaceholder")}
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
          {tn("capture.loanLine", {
            id: <strong>{loan.loan.loan_id}</strong>,
            purpose: <strong>{loan.loan.purpose ? purposeName(loan.loan.purpose) : "—"}</strong>,
          })}
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
              <img src={preview} alt={t("capture.photoAlt")} />
            </div>
          )}
          {busy && <BlockLoading text={t("capture.checking")} />}

          <div className="capture-actions">
            {camera === "on" ? (
              <button type="button" className="shutter" onClick={shoot} disabled={!canShoot}>
                {t("capture.shoot")}
              </button>
            ) : (
              <button
                type="button"
                className="shutter"
                onClick={startCamera}
                disabled={!canShoot || camera === "starting"}
              >
                {camera === "starting" ? t("capture.cameraStarting") : t("capture.cameraOn")}
              </button>
            )}
            {camera === "unavailable" && (
              <p className="error small" role="alert">
                {t("capture.cameraDenied")}
              </p>
            )}
            <label className={`secondary-file${canShoot ? "" : " disabled"}`}>
              <span className="muted small">{t("capture.orFile")}</span>
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
            {!loanId && ready && <p className="muted small">{t("capture.pickFirst")}</p>}
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
      ? [t("capture.geoNone"), ""]
      : r.geo_ok
        ? [t("capture.geoOk"), "ok"]
        : [t("capture.geoBad"), "warn"];
  const time =
    r.time_ok === null
      ? [t("capture.timeNone"), ""]
      : r.time_ok
        ? [t("capture.timeOk"), "ok"]
        : [t("capture.timeBad"), "warn"];
  const changed: [Color, Color] | null =
    before && after && before.color !== after.color ? [before.color, after.color] : null;
  const reasons = (after?.escalations ?? []).filter((e) => e.raises_color);

  return (
    <div className="cap-result" aria-live="polite">
      <h3 className="sub-h">{t("capture.result")}</h3>
      <div className="cap-row">
        <strong>{t("capture.assetType")}</strong>
        {am ? (
          <span>
            {localized(am.label_i18n, am.label_uz)}{" "}
            <span className="muted">{t("capture.score", { score: formatRisk(am.score) })}</span>
          </span>
        ) : (
          <span>{t("capture.assetUnknown")}</span>
        )}
      </div>
      {am && am.matches_purpose === false && (
        <p className="cap-warning" role="alert">
          {t("capture.assetMismatch")}
        </p>
      )}
      <div className={`cap-row ${r.duplicate_of.length ? "warn" : "ok"}`}>
        <strong>{t("capture.duplicate")}</strong>
        {r.duplicate_of.length ? (
          <span>
            {t("capture.dupFound")}{" "}
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
          <span>{t("capture.dupNone")}</span>
        )}
      </div>
      <div className={`cap-row ${geo[1]}`}>
        <strong>{t("capture.location")}</strong>
        <span>{geo[0]}</span>
      </div>
      <div className={`cap-row ${time[1]}`}>
        <strong>{t("capture.time")}</strong>
        <span>{time[0]}</span>
      </div>
      <div className="cap-row">
        <strong>{t("capture.privacy")}</strong>
        <span>
          {r.faces_blurred
            ? t("capture.facesBlurred", { count: typeof r.faces_count === "number" ? ` ${t("common.count", { n: r.faces_count })}` : "" })
            : t("capture.noFaces")}
        </span>
      </div>
      {changed && (
        <div className="cap-row warn">
          <strong>{t("capture.colorChanged")}</strong>
          <span className="cap-color">
            {COLOR_ICON[changed[0]]} {colorName(changed[0])} → {COLOR_ICON[changed[1]]} {colorName(changed[1])}
          </span>
          {reasons.map((e) => (
            <span key={e.code}>{escalationText(e)}</span>
          ))}
          <span className="muted small">{t("capture.priorityNote")}</span>
        </div>
      )}
      <div className="capture-actions">
        <a className="capture-link" href={loanHref(loanId)} onClick={(e) => onLinkClick(e, loanHref(loanId))}>
          {t("capture.openLoan")}
        </a>
        <button type="button" className="secondary" onClick={onAgain}>
          {t("capture.again")}
        </button>
      </div>
    </div>
  );
}
