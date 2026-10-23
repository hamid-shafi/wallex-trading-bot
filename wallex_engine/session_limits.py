"""Durable operator trial limits, independent of strategy signals."""
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import math
from pathlib import Path


MAX_SESSION_SECONDS = 7 * 24 * 3600
STOP_REASONS = frozenset(('operator_stop', 'capital_floor', 'capital_ceiling',
                          'time_limit', 'drawdown_halt'))


def _finite_number(value):
    # JSON booleans are integers in Python, but are never valid monetary values.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _timestamp(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'invalid {label}')
    return value


@dataclass(frozen=True)
class SessionLimits:
    floor_tmn: float | None = 4_000_000
    ceiling_tmn: float | None = 6_000_000
    seconds: int = 1800
    ends_at: int | None = None

    def _validate_settings(self):
        if any(v is not None and (not _finite_number(v) or v <= 0)
               for v in (self.floor_tmn, self.ceiling_tmn)):
            raise ValueError('invalid capital limit')
        if (self.floor_tmn is not None and self.ceiling_tmn is not None
                and self.floor_tmn >= self.ceiling_tmn):
            raise ValueError('capital floor must be below ceiling')
        if (isinstance(self.seconds, bool) or not isinstance(self.seconds, int)
                or not 60 <= self.seconds <= MAX_SESSION_SECONDS):
            raise ValueError('trial duration must be 60..604800 seconds')
        if self.ends_at is not None:
            _timestamp(self.ends_at, 'absolute deadline')

    def validate(self, initial):
        if not _finite_number(initial) or initial <= 0:
            raise ValueError('invalid initial capital')
        self._validate_settings()
        if ((self.floor_tmn is not None and self.floor_tmn >= initial) or
                (self.ceiling_tmn is not None and self.ceiling_tmn <= initial)):
            raise ValueError('initial cash must be strictly inside trial limits')
        return self

    def start(self, journal, now):
        self._validate_settings()
        _timestamp(now, 'session time')
        state = journal.get('trial')
        if state is None:
            deadline = self.ends_at if self.ends_at is not None else now + self.seconds
            if not now < deadline <= now + MAX_SESSION_SECONDS:
                raise ValueError('new session deadline must be in the next seven days')
            state = {'limits': asdict(self), 'started_at': now,
                     'deadline': deadline, 'reason': None}
            journal.put('trial', state)
        else:
            if not isinstance(state, dict) or state.get('limits') != asdict(self):
                raise ValueError('trial limits changed; explicit reconciliation required')
            started = _timestamp(state.get('started_at'), 'persisted session start')
            deadline = _timestamp(state.get('deadline'), 'persisted deadline')
            expected = self.ends_at if self.ends_at is not None else started + self.seconds
            if deadline != expected:
                raise ValueError('persisted deadline disagrees with trial limits')
            reason = state.get('reason')
            if reason is not None and (not isinstance(reason, str) or reason not in STOP_REASONS):
                raise ValueError('unrecognized persisted stop reason')
            if 'reason' not in state:
                raise ValueError('persisted stop state missing')
        return state

    def check(self, journal, equity, now, halted=False, manual=False):
        if not _finite_number(equity):
            raise ValueError('invalid liquidation equity')
        state = self.start(journal, now)
        reason = state['reason']
        if reason is None:
            reason = ('operator_stop' if manual else
                      'capital_floor' if self.floor_tmn is not None and equity <= self.floor_tmn else
                      'capital_ceiling' if self.ceiling_tmn is not None and equity >= self.ceiling_tmn else
                      'time_limit' if now >= state['deadline'] else
                      'drawdown_halt' if halted else None)
            if reason:
                state.update(reason=reason, triggered_at=now, trigger_equity_tmn=equity)
                journal.put('trial', state)
        return state


def load_operator_limits(seed, env):
    """Load optional capital bands and an explicitly timezone-qualified deadline."""
    path = Path(seed) / 'session-plan.json'
    plan = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
    if not isinstance(plan, dict):
        raise ValueError('session plan must be an object')
    if path.exists() and not {'floor_tmn', 'ceiling_tmn'} <= plan.keys():
        raise ValueError('session plan must explicitly declare both capital bounds')
    end = env.get('OPERATOR_END_AT', plan.get('ends_at'))
    if end is not None:
        if not isinstance(end, str) or not end:
            raise ValueError('deadline must be an ISO datetime string')
        try:
            value = datetime.fromisoformat(end)
        except ValueError as exc:
            raise ValueError('invalid ISO deadline') from exc
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('deadline timezone required')
        if value.microsecond:
            raise ValueError('deadline must use whole seconds')
        deadline = int(value.timestamp())
        seconds = MAX_SESSION_SECONDS
    else:
        deadline = None
        raw_seconds = env.get('OPERATOR_TRIAL_SECONDS', '1800')
        if isinstance(raw_seconds, bool) or not isinstance(raw_seconds, (str, int)):
            raise ValueError('trial duration must be an integer')
        try:
            seconds = int(raw_seconds)
        except ValueError as exc:
            raise ValueError('trial duration must be an integer') from exc
    limits = SessionLimits(floor_tmn=plan.get('floor_tmn', 4_000_000),
                           ceiling_tmn=plan.get('ceiling_tmn', 6_000_000),
                           seconds=seconds, ends_at=deadline)
    limits._validate_settings()
    return limits
