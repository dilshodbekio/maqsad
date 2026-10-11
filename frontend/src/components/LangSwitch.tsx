import { LANGS, setLang, useLang, t, type Lang } from "../i18n";

const LABEL: Record<Lang, string> = { uz: "UZ", en: "EN", ru: "RU" };

/** UZ | EN | RU switcher in the top-right of the header. */
export function LangSwitch() {
  const lang = useLang();
  return (
    <div className="lang-switch" role="group" aria-label={t("lang.label")}>
      {LANGS.map((l, i) => (
        <span key={l} className="lang-item">
          {i > 0 && <span className="lang-sep" aria-hidden="true">|</span>}
          <button
            type="button"
            className={`lang-btn${l === lang ? " active" : ""}`}
            aria-pressed={l === lang}
            lang={l}
            onClick={() => setLang(l)}
          >
            {LABEL[l]}
          </button>
        </span>
      ))}
    </div>
  );
}
