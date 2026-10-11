"""In-process queue for LLM draft reports (one replica).

Every LLM call goes through here, one at a time, at most `rpm` calls in any 60 s window (a sliding window, stricter than
a token bucket: no burst can exceed the provider's per-minute limit). A request that cannot be answered from the cache
gets a ticket (job_id) and polls it: queued (position, eta) -> running -> done | error.

- Same cache key => one task; every requester gets their own ticket (visible only to the sandbox that created it).
- A 429 from the provider (`LLMRateLimited`) keeps the task at the front of the queue and blocks the queue until the
  provider's retry time (at least 5 s, growing with repeated 429s, capped at 30 s). The user sees nothing but "queued".
  If the provider asks for longer than a job may wait (`job_timeout_s`, e.g. an exhausted DAILY quota: "retry in 1h49m"),
  waiting is pointless: every waiting task fails at once with the "busy" message and new requests get it until that time.
- A ticket older than `job_timeout_s` reports an error (Uzbek); the task itself keeps going and still fills the cache.
- User tasks (priority 0) go before pre-warm tasks (priority 1); a running pre-warm call is not interrupted.
- Jobs live in memory: after a restart job_id is unknown (404) and the client simply submits again (the cache answers).
"""
from __future__ import annotations

import itertools
import logging
import math
import threading
import time
import uuid
from collections import deque
from typing import Callable, Dict, List, Optional

from . import llm
from .db import utc_now
from .i18n import Msg

log = logging.getLogger("maqsad.draft_queue")

PRIORITY_USER, PRIORITY_PREWARM = 0, 1
MSG_BUSY = Msg("busy")
MSG_FAILED = Msg("draft_failed")
BACKOFF_MIN_S, BACKOFF_MAX_S = 5.0, 30.0
RECENT_DURATIONS = 20


class QueueFull(Exception):
    """Too many user tasks are waiting (the 'service is busy' message)."""


class Task:
    def __init__(self, seq: int, scope: str, loan_id: str, key: str, evidence: list, reasons: list, priority: int, lang: str = "uz"):
        self.seq, self.scope, self.loan_id, self.key, self.lang = seq, scope, loan_id, key, lang
        self.evidence, self.reasons, self.priority = evidence, reasons, priority
        self.status = "queued"  # queued | running | done | error
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.attempts = 0
        self.started: Optional[float] = None
        self.callbacks: List[Callable[["Task"], None]] = []


class Ticket:
    def __init__(self, job_id: str, task: Task, sandbox_id: str, created: float):
        self.job_id, self.task, self.sandbox_id, self.created = job_id, task, sandbox_id, created


class DraftQueue:
    def __init__(self, settings, db, *, rpm: int, job_timeout_s: float = 120.0, max_queue: int = 30,
                 ticket_ttl_s: float = 900.0, clock: Callable[[], float] = time.monotonic):
        self.settings, self.db = settings, db
        self.rpm, self.job_timeout_s, self.max_queue, self.ticket_ttl_s = max(1, rpm), job_timeout_s, max_queue, ticket_ttl_s
        self.clock = clock
        self._cond = threading.Condition()
        self._seq = itertools.count(1)
        self._tasks: Dict[tuple, Task] = {}  # (scope, loan_id, key) -> unfinished task
        self._queued: List[Task] = []
        self._running: Optional[Task] = None
        self._tickets: Dict[str, Ticket] = {}
        self._calls: deque = deque()  # start times of recent LLM calls (sliding window)
        self._blocked_until = 0.0
        self._quota_until = 0.0  # provider quota exhausted until then (retry_after > job_timeout_s)
        self._rate_hits = 0
        self._durations: deque = deque(maxlen=RECENT_DURATIONS)
        self._worker: Optional[threading.Thread] = None

    # ---------- submit / inspect ----------

    def submit(self, *, sandbox_id: str, scope: str, loan_id: str, key: str, evidence: list, reasons: list,
               priority: int = PRIORITY_USER, lang: str = "uz", cache_lookup: Optional[Callable[[], Optional[dict]]] = None,
               on_finish: Optional[Callable[[Task], None]] = None) -> Optional[Ticket]:
        """Returns a Ticket, or None when `cache_lookup` found a draft meanwhile (nothing to wait for).
        Raises QueueFull for a new user task when more than `max_queue` user tasks are waiting."""
        with self._cond:
            self._sweep()
            if cache_lookup is not None and cache_lookup() is not None:
                return None
            tkey = (scope, loan_id, key)
            task = self._tasks.get(tkey)
            if task is None:
                if self._quota_until > self.clock():
                    raise QueueFull()  # the provider will not answer within a job's lifetime
                if priority == PRIORITY_USER and sum(1 for t in self._queued if t.priority == PRIORITY_USER) >= self.max_queue:
                    raise QueueFull()
                task = Task(next(self._seq), scope, loan_id, key, evidence, reasons, priority, lang)
                self._tasks[tkey] = task
                self._queued.append(task)
                self._ensure_worker()
            elif priority < task.priority:
                task.priority = priority  # a user joined a pre-warm task: it moves up
            if on_finish is not None:
                task.callbacks.append(on_finish)
            ticket = Ticket(uuid.uuid4().hex, task, sandbox_id, self.clock())
            self._tickets[ticket.job_id] = ticket
            self._cond.notify_all()
            return ticket

    def view(self, job_id: str, sandbox_id: str) -> Optional[dict]:
        """Job state for its owner sandbox; None for unknown ids and for other sandboxes' jobs."""
        with self._cond:
            ticket = self._tickets.get(job_id)
            if ticket is None or ticket.sandbox_id != sandbox_id:
                return None
            return self._view(ticket)

    def ticket_view(self, ticket: Ticket) -> dict:
        with self._cond:
            return self._view(ticket)

    def _view(self, ticket: Ticket) -> dict:
        t, now = ticket.task, self.clock()
        out: dict = {"job_id": ticket.job_id, "status": t.status, "position": 0, "eta_s": None}
        if t.status == "done":
            out["result"] = t.result
            return out
        if t.status == "error":
            out["error"] = t.error
            return out
        if now - ticket.created > self.job_timeout_s:
            out.update(status="error", error=MSG_BUSY)
            return out
        if t.status == "running":
            avg = self._avg_duration()
            if avg is not None and t.started is not None:
                out["eta_s"] = max(0, math.ceil(avg - (now - t.started)))
            return out
        ahead = sum(1 for x in self._queued if (x.priority, x.seq) < (t.priority, t.seq))
        out["position"] = ahead + 1 + (1 if self._running is not None else 0)
        out["eta_s"] = self._eta(now, out["position"])
        return out

    def stats(self) -> dict:
        with self._cond:
            return {"queued": len(self._queued), "running": self._running is not None,
                    "calls_last_minute": self._calls_in_window(self.clock())}

    # ---------- timing ----------

    def _avg_duration(self) -> Optional[float]:
        return sum(self._durations) / len(self._durations) if self._durations else None

    def _calls_in_window(self, now: float) -> int:
        return sum(1 for c in self._calls if now - c < 60.0)

    def _slot_wait(self, now: float) -> float:
        """Seconds until the next call may start (sliding window + provider back-off)."""
        wait = max(0.0, self._blocked_until - now)
        recent = [c for c in self._calls if now - c < 60.0]
        if len(recent) >= self.rpm:
            wait = max(wait, recent[-self.rpm] + 60.0 - now)
        return wait

    def _eta(self, now: float, position: int) -> Optional[int]:
        """Estimate: the `position`-th call from now. Start times follow the sliding window; each call lasts the average of
        the recent ones (unknown until one has finished: then only the window spacing is counted)."""
        avg = self._avg_duration() or 0.0
        starts = [c for c in self._calls if now - c < 60.0]
        t = now + max(0.0, self._blocked_until - now)
        for _ in range(position):
            if len(starts) >= self.rpm:
                t = max(t, starts[-self.rpm] + 60.0)
            starts.append(t)
            t += avg
        return max(0, math.ceil(t - now))

    # ---------- worker ----------

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="draft-queue", daemon=True)
            self._worker.start()

    def _next_task(self) -> Task:
        with self._cond:
            while True:
                if not self._queued:
                    self._cond.wait()
                    continue
                now = self.clock()
                wait = self._slot_wait(now)
                if wait > 0:
                    self._cond.wait(timeout=wait)  # re-evaluated: a user task may have arrived meanwhile
                    continue
                task = min(self._queued, key=lambda t: (t.priority, t.seq))
                self._queued.remove(task)
                task.status, task.started, task.attempts = "running", now, task.attempts + 1
                self._running = task
                self._calls.append(now)
                while self._calls and now - self._calls[0] > 120.0:
                    self._calls.popleft()
                return task

    def _run(self) -> None:
        while True:
            task = self._next_task()
            t0 = self.clock()
            try:
                result = llm.draft_report(self.settings, task.loan_id, task.evidence, task.reasons, self.db, lang=task.lang)
                result["generated_at"] = utc_now()
                self.db.draft_cache_put(task.scope, task.loan_id, task.key, result, task.lang)
                self._finish(task, result=result, duration=self.clock() - t0)
            except llm.LLMRateLimited as e:
                self._requeue_after_429(task, e.retry_after_s)
            except Exception as e:  # noqa: BLE001 - llm.LLMError and anything unexpected: the job fails, the worker lives on
                log.warning("draft task failed loan=%s: %s", task.loan_id, e)
                self._finish(task, error=MSG_FAILED)

    def _requeue_after_429(self, task: Task, retry_after_s: float) -> None:
        if retry_after_s > self.job_timeout_s:
            self._quota_exhausted(task, retry_after_s)
            return
        with self._cond:
            self._rate_hits += 1
            delay = max(retry_after_s, min(BACKOFF_MAX_S, BACKOFF_MIN_S * self._rate_hits))
            self._blocked_until = self.clock() + delay
            task.status = "queued"
            self._queued.append(task)  # same seq: it stays ahead of later tasks of its priority
            self._running = None
            log.warning("LLM 429: queue paused for %.1fs (loan=%s, attempt %d)", delay, task.loan_id, task.attempts)
            self._cond.notify_all()

    def _quota_exhausted(self, task: Task, retry_after_s: float) -> None:
        with self._cond:
            self._quota_until = self.clock() + retry_after_s
            doomed = [task] + self._queued
            self._queued = []
            for t in doomed:
                self._tasks.pop((t.scope, t.loan_id, t.key), None)
            log.error("LLM quota exhausted (retry in %.0fs): %d draft task(s) failed, new ones are refused until then",
                      retry_after_s, len(doomed))
            self._running = None
        for t in doomed:
            self._finish(t, error=MSG_BUSY, requeue_check=False)

    def _finish(self, task: Task, *, result: Optional[dict] = None, error: Optional[str] = None,
                duration: Optional[float] = None, requeue_check: bool = True) -> None:
        with self._cond:
            if result is not None:
                task.result, task.status = {**result, "cached": False}, "done"
                self._rate_hits = 0
                if duration is not None:
                    self._durations.append(duration)
            else:
                task.error, task.status = error, "error"
            self._tasks.pop((task.scope, task.loan_id, task.key), None)
            if requeue_check:
                self._running = None
            callbacks, task.callbacks = task.callbacks, []
            self._cond.notify_all()
        for cb in callbacks:
            try:
                cb(task)
            except Exception:  # noqa: BLE001
                log.exception("draft task callback failed")

    def _sweep(self) -> None:
        now = self.clock()
        for job_id in [j for j, t in self._tickets.items()
                       if now - t.created > self.ticket_ttl_s and t.task.status in ("done", "error")]:
            del self._tickets[job_id]

    def forget_tickets(self) -> None:
        """Admin reset: every job id becomes unknown (clients resubmit and get the cache or a new job)."""
        with self._cond:
            self._tickets.clear()


class Prewarm:
    """Fills the shared cache for the top-N queue loans through the same queue, at the lowest priority."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total = self.done = self.skipped = self.failed = 0
        self._pending = 0
        self._thread: Optional[threading.Thread] = None

    def status(self) -> dict:
        with self._lock:
            running = (self._thread is not None and self._thread.is_alive()) or self._pending > 0
            return {"total": self.total, "done": self.done, "skipped": self.skipped, "failed": self.failed,
                    "running": running}

    def start(self, loan_ids: Callable[[], List[str]], handle: Callable[[str, Callable[[Task], None]], str]) -> bool:
        """`handle(loan_id, on_finish)` returns 'cached' | 'queued'. False if a run is still going."""
        with self._lock:
            if (self._thread is not None and self._thread.is_alive()) or self._pending > 0:
                return False
            self.total = self.done = self.skipped = self.failed = 0
            self._thread = threading.Thread(target=self._run, args=(loan_ids, handle), name="prewarm", daemon=True)
            self._thread.start()
            return True

    def _run(self, loan_ids, handle) -> None:
        try:
            ids = loan_ids()
        except Exception:  # noqa: BLE001
            log.exception("prewarm: cannot list loans")
            return
        with self._lock:
            self.total = len(ids)
        for lid in ids:
            try:
                with self._lock:
                    self._pending += 1
                outcome = handle(lid, self._on_finish)
                if outcome == "cached":
                    with self._lock:
                        self._pending -= 1
                        self.skipped += 1
            except Exception:  # noqa: BLE001
                log.exception("prewarm: loan %s failed", lid)
                with self._lock:
                    self._pending -= 1
                    self.failed += 1

    def _on_finish(self, task: Task) -> None:
        with self._lock:
            self._pending -= 1
            if task.status == "done":
                self.done += 1
            else:
                self.failed += 1
