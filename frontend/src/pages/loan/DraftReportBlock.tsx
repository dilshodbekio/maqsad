import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type DraftJob, type DraftReport } from "../../api";
import { formatLocalDateTime } from "../../format";
import { t } from "../../i18n";
import { BlockError, BlockLoading, EvidenceChip } from "./evidence";

type State =
  | { kind: "idle" }
  | { kind: "loading" } // the request is on its way
  | { kind: "waiting"; job: DraftJob } // queued / running on the server
  | { kind: "unavailable"; detail: string }
  | { kind: "error"; message: string }
  | { kind: "ok"; data: DraftReport };

const POLL_MS = 2000;
const REGENERATE_COOLDOWN_S = 60;
const jobKey = (loanId: string) => `maqsad:draft-job:${loanId}`;
const REGEN_KEY = "maqsad:draft-regenerated-at";

// sessionStorage may be unavailable (private mode): everything still works, only resuming / the cooldown memory is lost
function store(key: string, value: string | null): void {
  try {
    if (value === null) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, value);
  } catch {
    // ignore
  }
}
function stored(key: string): string | null {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

const isJob = (x: DraftReport | DraftJob): x is DraftJob => "job_id" in x;

function cooldownLeft(): number {
  const at = Number(stored(REGEN_KEY));
  if (!at) return 0;
  return Math.max(0, Math.ceil(REGENERATE_COOLDOWN_S - (Date.now() - at) / 1000));
}

/** LLM draft of the inspection report. Every fact is linked to its evidence; unverified facts are kept, marked.
 * The server answers from its cache at once; otherwise it queues the job (the LLM has a per-minute limit) and we show
 * the place in the queue, polling until the draft is ready. The job id is kept in sessionStorage, so leaving the page
 * and coming back continues the same job. */
export function DraftReportBlock({ loanId }: { loanId: string }) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [cooldown, setCooldown] = useState(cooldownLeft);
  const alive = useRef(true);
  const timer = useRef<number | undefined>(undefined);
  const run = useRef(0); // a newer request supersedes the polling of an older one

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      window.clearTimeout(timer.current);
    };
  }, []);

  useEffect(() => {
    if (cooldown <= 0) return;
    const t = window.setTimeout(() => setCooldown(cooldownLeft()), 1000);
    return () => window.clearTimeout(t);
  }, [cooldown]);

  const current = (id: number) => alive.current && id === run.current;

  const fail = useCallback(
    (e: Error, id: number) => {
      if (!current(id)) return;
      store(jobKey(loanId), null);
      setState(
        e instanceof ApiError && e.status === 503
          ? { kind: "unavailable", detail: e.message }
          : { kind: "error", message: e.message },
      );
    },
    [loanId],
  );

  // submit (or re-submit) the request; the server returns the draft (cache) or a job to follow
  const submit = useCallback(
    (regenerate: boolean, id: number) => {
      // eslint-disable-next-line @typescript-eslint/no-use-before-define
      api.draftReport(loanId, regenerate).then((x) => (isJob(x) ? follow(x, id) : done(x, id))).catch((e: Error) => fail(e, id));
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [loanId, fail],
  );

  const done = (data: DraftReport, id: number) => {
    if (!current(id)) return;
    store(jobKey(loanId), null);
    setState({ kind: "ok", data });
  };

  // show the job's state and poll it every POLL_MS until it ends
  const follow = (job: DraftJob, id: number) => {
    if (!current(id)) return;
    if (job.status === "done" && job.result) return done(job.result, id);
    if (job.status === "error") {
      store(jobKey(loanId), null);
      setState({ kind: "unavailable", detail: job.error ?? t("report.failed") });
      return;
    }
    store(jobKey(loanId), job.job_id);
    setState({ kind: "waiting", job });
    timer.current = window.setTimeout(() => {
      api
        .draftJob(job.job_id)
        .then((j) => follow(j, id))
        .catch((e: Error) => {
          if (!current(id)) return;
          if (e instanceof ApiError && e.status === 404) submit(false, id); // the server lost the job: ask again
          else if (e instanceof ApiError && e.status !== 0) fail(e, id);
          else follow(job, id); // connection hiccup: keep waiting, no error shown
        });
    }, POLL_MS);
  };

  const start = (regenerate = false) => {
    const id = ++run.current;
    window.clearTimeout(timer.current);
    setState({ kind: "loading" });
    if (regenerate) {
      store(REGEN_KEY, String(Date.now()));
      setCooldown(REGENERATE_COOLDOWN_S);
    }
    submit(regenerate, id);
  };

  // came back to the page while a job was running: continue with its job_id
  useEffect(() => {
    const jobId = stored(jobKey(loanId));
    if (!jobId) return;
    const id = ++run.current;
    setState({ kind: "loading" });
    api
      .draftJob(jobId)
      .then((job) => follow(job, id))
      .catch((e: Error) => {
        if (!current(id)) return;
        store(jobKey(loanId), null);
        if (e instanceof ApiError && e.status === 404) submit(false, id);
        else setState({ kind: "idle" });
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loanId]);

  const busy = state.kind === "loading" || state.kind === "waiting";

  return (
    <section className="card" aria-labelledby="h-report">
      <h2 id="h-report">{t("report.title")}</h2>
      {state.kind !== "ok" && !busy && (
        <button type="button" onClick={() => start()}>
          {t("report.start")}
        </button>
      )}
      {state.kind === "idle" && (
        <p className="muted small">{t("report.hint")}</p>
      )}
      {state.kind === "loading" && <BlockLoading text={t("report.requesting")} />}
      {state.kind === "waiting" && <Waiting job={state.job} />}
      {state.kind === "unavailable" && (
        <div className="notice" role="status">
          <p>
            <strong>{t("report.unavailable")}</strong>
          </p>
          <p className="muted small">{state.detail}</p>
        </div>
      )}
      {state.kind === "error" && <BlockError message={t("report.error", { message: state.message })} onRetry={() => start()} />}
      {state.kind === "ok" && <Report data={state.data} cooldown={cooldown} onRefresh={() => start(true)} />}
    </section>
  );
}

function Waiting({ job }: { job: DraftJob }) {
  const text =
    job.status === "queued"
      ? t("report.queued", {
          position: job.position,
          eta: job.eta_s != null && job.eta_s > 0 ? t("report.eta", { seconds: job.eta_s }) : "",
        })
      : t("report.running");
  return (
    <div aria-live="polite">
      <BlockLoading text={text} />
      {job.status === "queued" && (
        <p className="muted small">{t("report.queueNote")}</p>
      )}
    </div>
  );
}

function Report({ data, cooldown, onRefresh }: { data: DraftReport; cooldown: number; onRefresh: () => void }) {
  const verified = data.facts.filter((f) => !f.unverified);
  const unverified = data.facts.filter((f) => f.unverified);
  return (
    <div className="report">
      <p className="report-badge small">{t("report.badge")}</p>
      <div className="report-text">
        {data.text_uz
          .split(/\n+/)
          .filter((p) => p.trim())
          .map((p, i) => (
            <p key={i}>{p}</p>
          ))}
      </div>

      {verified.length > 0 && (
        <>
          <h3 className="sub-h">{t("report.verified")}</h3>
          <ul className="facts">
            {verified.map((f, i) => (
              <li key={i}>
                <span>{f.claim}</span> <EvidenceChip refId={f.evidence_ref} />
              </li>
            ))}
          </ul>
        </>
      )}
      {unverified.length > 0 && (
        <>
          <h3 className="sub-h warn">{t("report.unverified", { n: unverified.length })}</h3>
          <p className="muted small">{t("report.unverifiedNote")}</p>
          <ul className="facts unverified">
            {unverified.map((f, i) => (
              <li key={i}>
                <span>{f.claim}</span> {f.evidence_ref ? <EvidenceChip refId={f.evidence_ref} /> : null}
              </li>
            ))}
          </ul>
        </>
      )}
      {data.facts.length === 0 && <p className="muted small">{t("report.noFacts")}</p>}

      <p className="report-meta">
        {t("app.footerModel", { version: data.model })} · {data.cached ? t("report.metaCached") : t("report.metaNew")}
        {data.generated_at ? t("report.metaCreated", { time: formatLocalDateTime(data.generated_at) }) : ""} ·{" "}
        {data.cached ? t("report.metaRead") : t("report.metaLatency")} {Math.round(data.latency_ms)} ms · {t("report.metaCost")}{" "}
        {data.cost_usd === null ? t("report.costUnknown") : data.cost_usd === 0 ? t("report.costZero") : `$${data.cost_usd.toFixed(5)}`}
        {data.input_tokens != null && data.output_tokens != null
          ? t("report.tokens", { in: data.input_tokens, out: data.output_tokens })
          : ""}
      </p>
      <button type="button" className="secondary" onClick={onRefresh} disabled={cooldown > 0}>
        {cooldown > 0 ? t("report.regenerateWait", { seconds: cooldown }) : t("report.regenerate")}
      </button>
    </div>
  );
}
