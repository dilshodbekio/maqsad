import { useState } from "react";
import { api, type Duplicate, type LoanDetail, type Photo } from "../../api";
import { formatDateTime, formatSimilarity } from "../../format";
import { t } from "../../i18n";
import { onLinkClick } from "../../router";
import { BlockError, BlockLoading, duplicatesOf, field, isSandboxPhotoRef, useFocus, type Load } from "./evidence";

const loanHref = (id: string) => `/loan/${encodeURIComponent(id)}`;

/** Portfolio photos (ml/assets via /api/asset) and this sandbox's uploads, side by side. */
export function Photos({ detail }: { detail: LoanDetail }) {
  const portfolio = detail.photos.filter((p) => p.source === "portfolio");
  const uploads = detail.photos.filter((p) => p.source === "borrower_upload");
  const { activeRef } = useFocus();
  const uploadsActive = activeRef !== null && isSandboxPhotoRef(activeRef);

  return (
    <section className="card" aria-labelledby="h-photos">
      <h2 id="h-photos">{t("photos.title")}</h2>
      {detail.photos.length === 0 && <p className="muted">{t("photos.none")}</p>}
      <div className="photo-columns">
        {portfolio.length > 0 && (
          <div>
            <h3 className="sub-h">{t("photos.portfolio")}</h3>
            <div className="photo-grid">
              {portfolio.map((p) => (
                <PhotoCard key={p.photo_id} photo={p} detail={detail} />
              ))}
            </div>
          </div>
        )}
        {uploads.length > 0 && (
          <div id="photos-uploads" className={uploadsActive ? "is-active-group" : undefined}>
            <h3 className="sub-h">{t("photos.uploads")}</h3>
            <div className="photo-grid">
              {uploads.map((p) => (
                <PhotoCard key={p.photo_id} photo={p} detail={detail} />
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function okText(ok: boolean | null | undefined, yes: string, no: string): { text: string; cls: string } {
  if (ok === true) return { text: yes, cls: "ok" };
  if (ok === false) return { text: no, cls: "warn" };
  return { text: t("common.unknown"), cls: "muted" };
}

function PhotoCard({ photo, detail }: { photo: Photo; detail: LoanDetail }) {
  const { activeRef, evidence } = useFocus();
  const [imgFailed, setImgFailed] = useState(false);
  const upload = photo.source === "borrower_upload";
  const ev = upload ? undefined : evidence.get(`photo:${photo.photo_id}`);

  const dups: Duplicate[] = upload
    ? photo.check?.duplicate_of ?? []
    : photo.duplicate_of.length
      ? photo.duplicate_of
      : duplicatesOf(ev);
  const geoOk = upload ? photo.check?.geo_ok : field<boolean | null>(ev, "geo_ok");
  const timeOk = upload ? photo.check?.time_ok : field<boolean | null>(ev, "time_ok");
  const distance = field<number>(ev, "distance_km");
  const geo = okText(geoOk, t("photos.geoOk"), t("photos.geoBad"));
  const time = okText(timeOk, t("photos.timeOk"), t("photos.timeBad"));
  const active = activeRef === `photo:${photo.photo_id}` || (upload && activeRef !== null && isSandboxPhotoRef(activeRef));

  return (
    <figure id={`photo-${photo.photo_id}`} className={`photo-card${active ? " is-active" : ""}`}>
      {photo.url && !imgFailed ? (
        <img
          src={photo.url}
          alt={upload ? t("photos.imgBorrower") : t("photos.imgPortfolio")}
          loading="lazy"
          onError={() => setImgFailed(true)}
        />
      ) : (
        <div className="photo-missing">{photo.url ? t("photos.imgFailed") : t("photos.imgNone")}</div>
      )}
      <figcaption className="small">
        <div className="muted">
          {upload ? t("photos.uploaded") : t("photos.taken")}: {formatDateTime(photo.captured_at)}
          {upload && photo.client_captured_at && t("photos.deviceTime", { time: formatDateTime(photo.client_captured_at) })}
        </div>
        <div className="photo-flags">
          <span className={geo.cls}>
            📍 {geo.text}
            {geoOk === false && typeof distance === "number" && ` (${distance.toFixed(1)} km)`}
          </span>
          <span className={time.cls}>🕒 {time.text}</span>
          {upload && photo.check && (
            <span className="muted">
              {photo.check.faces_blurred
                ? t("photos.facesBlurred", { count: photo.check.faces_count ? ` ${t("common.count", { n: photo.check.faces_count })}` : "" })
                : t("photos.noFaces")}
            </span>
          )}
        </div>
        {dups.length > 0 ? (
          <div className="dup-box">
            <strong className="warn">{t("photos.dup")}</strong>
            <ul>
              {dups.map((d) => (
                <DuplicateRow key={d.loan_id} dup={d} self={detail.loan.loan_id} />
              ))}
            </ul>
          </div>
        ) : (
          <div className="ok">{t("photos.noDup")}</div>
        )}
      </figcaption>
    </figure>
  );
}

/** "L1504307 kreditidagi rasm bilan" + on demand that loan's photo next to this one. */
function DuplicateRow({ dup, self }: { dup: Duplicate; self: string }) {
  const [other, setOther] = useState<Load<Photo[]> | null>(null);
  const load = () => {
    setOther({ kind: "loading" });
    api
      .loan(dup.loan_id)
      .then((d) => setOther({ kind: "ok", data: d.photos.filter((p) => p.url) }))
      .catch((e: Error) => setOther({ kind: "error", message: e.message }));
  };
  if (dup.loan_id === self) return null;
  return (
    <li>
      <a href={loanHref(dup.loan_id)} onClick={(e) => onLinkClick(e, loanHref(dup.loan_id))}>
        {dup.loan_id}
      </a>{" "}
      {t("photos.dupRow", { similarity: formatSimilarity(dup.similarity) })}{" "}
      {other === null && (
        <button type="button" className="link-button" onClick={load}>
          {t("photos.compare")}
        </button>
      )}
      {other?.kind === "loading" && <BlockLoading text={t("common.loading")} />}
      {other?.kind === "error" && <BlockError message={t("photos.compareError", { message: other.message })} onRetry={load} />}
      {other?.kind === "ok" &&
        (other.data.length ? (
          <div className="dup-compare">
            {other.data.slice(0, 2).map((p) => (
              <img key={p.photo_id} src={p.url!} alt={t("photos.imgOfLoan", { id: dup.loan_id })} loading="lazy" />
            ))}
          </div>
        ) : (
          <span className="muted">{t("photos.noImgOfLoan")}</span>
        ))}
    </li>
  );
}
