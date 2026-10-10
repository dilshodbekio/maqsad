import { goBack, onLinkClick } from "../router";

// Placeholder until the loan page (reasons, money trail, photos, report, decision) is built.
export function LoanPlaceholder({ loanId }: { loanId: string }) {
  return (
    <section className="card">
      <a
        href="/"
        className="back-link"
        onClick={(e) => {
          if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
          e.preventDefault();
          goBack("/");
        }}
      >
        ← Navbatga qaytish
      </a>
      <h2>Kredit {loanId}</h2>
      <p className="muted">Kredit sahifasi keyingi bosqichda tayyorlanadi: sabablar, pul izi, rasmlar, faktura va qaror.</p>
      <p>
        <a href="/" onClick={(e) => onLinkClick(e, "/")}>
          Navbat boshiga o'tish
        </a>
      </p>
    </section>
  );
}
