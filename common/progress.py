"""Live stage of an agent run for job polling: a forward-only state machine with reader-facing labels.

The job pool binds a Progress to each job's context (`bind`); the pipeline reports through the module functions
(`enter`, `tick`, `track`/`phase`, `branch`), which are no-ops outside a job (CLI, tests). Parallel tasks copy the
context, so every branch and `Send` task reports into the same Progress."""
import contextvars
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

# Order is the machine: a run only moves forward along it; failed/interrupted are terminal from any stage.
STAGES = {
    'queued': 'В очереди',
    'preparing_reports': 'Готовим отчёты банка',
    'collecting': 'Собираем факты о компании и подбираем новости',
    'analyzing': 'Разбираем новости',
    'generating': 'Формируем инсайты',
    'verifying': 'Проверяем инсайты',
    'finalizing': 'Ещё чуть-чуть: собираем итог',
    'done': 'Готово',
}
TERMINAL = {'done': 'Готово', 'failed': 'Не получилось', 'interrupted': 'Прервано: перезапустите анализ'}
ORDER = list(STAGES)
DEFAULT_PLAN = ['collecting', 'analyzing', 'generating', 'verifying', 'finalizing']
BRANCHES = {'analytics': 'Разбираем аналитику банка'}
# per-insight phases inside generating/verifying; a checker's rewrite goes back to «write» without moving the stage back
PHASES = {'judge': 'generating', 'write': 'generating', 'check': 'verifying', 'audit': 'verifying', 'done': None}

_current: contextvars.ContextVar['Progress | None'] = contextvars.ContextVar('job_progress', default=None)
_item: contextvars.ContextVar[object] = contextvars.ContextVar('job_progress_item', default=None)


def _now():
    return datetime.now(timezone.utc).isoformat()


class Progress:
    """Thread-safe stage of one job; `on_change(snapshot)` gets every new state with an increasing `seq`."""

    def __init__(self, on_change=None, plan=None, on_results=None):
        self.lock = threading.Lock()
        self.on_change = on_change
        self.on_results = on_results
        self.on_event = None
        self.plan_codes = list(plan or DEFAULT_PLAN)
        self.code = 'queued'
        self.label = None
        self.done = self.total = None
        self.items: dict = {}
        self.branches: dict = {}
        self.history = [dict(code='queued', label=STAGES['queued'], at=_now())]
        self.seq = 0
        self.updated_at = self.history[0]['at']

    # --- transitions (each returns the new snapshot to publish, or None when nothing changed)
    def _enter(self, code, label=None, total=None):
        if self.code in TERMINAL or ORDER.index(code) <= ORDER.index(self.code):
            return False
        self.code, self.label = code, label
        self.done, self.total = (0, total) if total is not None else (None, None)
        self.history.append(dict(code=code, label=label or STAGES[code], at=_now()))
        return True

    def plan(self, codes):
        def step():
            self.plan_codes = [c for c in ORDER if c in codes]
            return True
        return self._apply(step)

    def enter(self, code, label=None, total=None):
        return self._apply(lambda: self._enter(code, label, total))

    def tick(self, code):
        def step():
            if self.code != code or self.total is None:
                return False
            self.done = min(self.done + 1, self.total)
            return True
        return self._apply(step)

    def phase(self, key, name):
        def step():
            if self.code in TERMINAL or ORDER.index(self.code) > ORDER.index('verifying'):
                return False
            self.items[key] = name
            open_ = [PHASES[p] for p in self.items.values() if PHASES[p]]
            target = 'generating' if 'generating' in open_ else 'verifying' if open_ else None
            changed = target is not None and self._enter(target, total=len(self.items))
            if self.code in ('generating', 'verifying'):
                done = sum(p == 'done' for p in self.items.values())
                changed = changed or done != self.done or self.total != len(self.items)
                self.done, self.total = done, max(len(self.items), self.total or 0)
            return changed
        return self._apply(step)

    def expect(self, n):
        """The number of insights the generating stage will work on (known when ranking ends); keys 0..n-1."""
        def step():
            for i in range(n):
                self.items.setdefault(i, 'judge')
            return self._enter('generating', total=n) if n else False
        return self._apply(step)

    def branch(self, name, status, error=None):
        def step():
            self.branches[name] = dict(status=status, label=BRANCHES.get(name, name), **({'error': error} if error else {}))
            return True
        return self._apply(step)

    def finish(self, code, error=None):
        def step():
            if self.code in TERMINAL:
                return False
            self.code, self.label = code, None
            if code == 'done' and self.total is not None:
                self.done = self.total
            self.history.append(dict(code=code, label=TERMINAL[code], at=_now(), **({'error': error} if error else {})))
            return True
        return self._apply(step)

    def _apply(self, step):
        with self.lock:
            if not step():
                return None
            self.seq += 1
            self.updated_at = _now()
            snap = self._snapshot()
        if self.on_change is not None:
            self.on_change(snap)
        return snap

    # --- what the job API shows
    def _label(self):
        if self.label:
            return self.label
        if self.code in TERMINAL:
            return TERMINAL[self.code]
        analytics = self.branches.get('analytics', {}).get('status') == 'running'
        if self.code == 'analyzing' and analytics:
            return 'Разбираем новости и аналитику банка'
        if self.code in ('generating', 'verifying') and analytics and self.total and self.done == self.total:
            return 'Ещё чуть-чуть: дожидаемся аналитики банка'
        return STAGES[self.code]

    def _snapshot(self):
        plan = self.plan_codes
        step = plan.index(self.code) + 1 if self.code in plan else (len(plan) if self.code == 'done' else 0)
        return dict(code=self.code, label=self._label(), step=step, steps=len(plan),
                    done=self.done, total=self.total,
                    branches={k: dict(v) for k, v in self.branches.items()},
                    started_at=self.history[0]['at'], updated_at=self.updated_at, seq=self.seq,
                    history=[dict(h) for h in self.history])

    def snapshot(self):
        with self.lock:
            return self._snapshot()


def current() -> Progress | None:
    return _current.get()


def listen(listener):
    """Use the same job context for checked-card events and stage updates. CLI has no listener."""
    if (p := _current.get()) is not None:
        p.on_event = listener


def emit(event, **data):
    if (p := _current.get()) is not None and p.on_event is not None:
        try:
            p.on_event(event, **data)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception('Live insight listener failed: %s', exc)


def results(rows):
    if (p := _current.get()) is not None and p.on_results is not None:
        p.on_results(rows)


def bind(progress: Progress | None):
    """Bind for the rest of the current context (the job's own copied context)."""
    _current.set(progress)


def plan(codes):
    if (p := _current.get()) is not None:
        p.plan(codes)


def enter(code, label=None, total=None):
    if (p := _current.get()) is not None:
        p.enter(code, label, total)


def tick(code):
    if (p := _current.get()) is not None:
        p.tick(code)


def expect(n):
    if (p := _current.get()) is not None:
        p.expect(n)


def branch(name, status, error=None):
    if (p := _current.get()) is not None:
        p.branch(name, status, error)


@contextmanager
def track(key):
    """Phases reported inside belong to insight `key` (one `Send` task); it is done when the block exits."""
    token = _item.set(key)
    try:
        yield
    finally:
        phase('done')
        _item.reset(token)


def phase(name):
    p, key = _current.get(), _item.get()
    if p is not None and key is not None:
        p.phase(key, name)
