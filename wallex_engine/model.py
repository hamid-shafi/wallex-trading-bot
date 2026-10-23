"""Online recursive ridge ensemble. All training targets mature before use."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import statistics

NAMES = ["bias", "r1", "r3", "r12", "r36", "z24", "trend", "vol_ratio",
         "range", "body", "volume", "trend_x_momentum", "reversion_x_range"]


def clip(x, limit=4.0):
    return max(-limit, min(limit, x))


def features(bars, horizon):
    if len(bars) < 65:
        return None
    b = bars[-65:]
    returns = [math.log(b[i].close / b[i-1].close) for i in range(1, len(b))]
    vol = max(statistics.pstdev(returns[-24:]), .0003)
    long_vol = max(statistics.pstdev(returns), .0003)
    closes = [x.close for x in b]
    mean = statistics.fmean(closes[-24:])
    sd = max(statistics.pstdev(closes[-24:]), b[-1].close * .0003)
    z = (closes[-1] - mean) / sd
    trend = math.log(statistics.fmean(closes[-8:]) / statistics.fmean(closes[-36:])) / (vol * 6)
    atr = statistics.fmean(max(b[i].high-b[i].low, abs(b[i].high-b[i-1].close),
                              abs(b[i].low-b[i-1].close)) for i in range(len(b)-14, len(b)))
    mom = sum(returns[-12:]) / (vol * math.sqrt(12))
    relative_vol = b[-1].volume / max(statistics.fmean(x.volume for x in b[-25:-1]), 1e-12)
    raw = [1, returns[-1] / vol, sum(returns[-3:]) / (vol * math.sqrt(3)), mom,
           sum(returns[-36:]) / (vol * 6), z, trend, vol / long_vol - 1,
           (b[-1].high-b[-1].low) / max(atr, 1e-12) - 1,
           math.log(b[-1].close/b[-1].open) / vol, math.log(max(relative_vol, 1e-8)),
           trend * mom, -z * max(0, 1-abs(trend))]
    return [clip(v) for v in raw], vol * math.sqrt(horizon), atr


class Ridge:
    def __init__(self, forgetting, n=len(NAMES), ridge=20):
        self.forgetting = forgetting
        self.w = [0.0] * n
        self.p = [[(1 / ridge if i == j else 0.0) for j in range(n)] for i in range(n)]
        self.mse = 1.0
        self.samples = 0

    def predict(self, x):
        px = [sum(a*b for a, b in zip(row, x)) for row in self.p]
        leverage = max(0, sum(a*b for a, b in zip(x, px)))
        return sum(a*b for a, b in zip(self.w, x)), math.sqrt(self.mse * leverage)

    def learn(self, x, target):
        pred, _ = self.predict(x)
        error = clip(target - pred, 5)
        px = [sum(a*b for a, b in zip(row, x)) for row in self.p]
        denom = self.forgetting + sum(a*b for a, b in zip(x, px))
        gain = [v / denom for v in px]
        self.w = [w + k * error for w, k in zip(self.w, gain)]
        self.p = [[(self.p[i][j] - gain[i]*px[j]) / self.forgetting
                   for j in range(len(x))] for i in range(len(x))]
        self.mse = .99 * self.mse + .01 * min(25, error*error)
        self.samples += 1


@dataclass
class Forecast:
    mu: float
    uncertainty: float
    lower: float
    atr: float
    samples: int
    disagreement: float
    contributions: dict
    model_hash: str


class Learner:
    def __init__(self, horizon, penalty):
        self.horizon, self.penalty = horizon, penalty
        self.models = [Ridge(v) for v in (.995, .999, 1.0)]
        self.pending = deque()
        self.history = []
        self.last_ts = 0

    def observe(self, bar):
        if bar.ts <= self.last_ts:
            raise ValueError("out-of-order or duplicate observation")
        self.last_ts = bar.ts
        # The entry price of a decision at t is open[t+1], never close[t].
        for item in self.pending:
            if item[4] is None:
                item[4] = bar.open
        self.history.append(bar)
        index = len(self.history)-1
        while self.pending and self.pending[0][0] + self.horizon <= index:
            _, x, scale, _, entry = self.pending.popleft()
            target = clip(math.log(bar.close / entry) / scale, 5)
            for m in self.models:
                m.learn(x, target)
        f = features(self.history, self.horizon)
        if f is None:
            return None
        x, scale, atr = f
        self.pending.append([index, x, scale, bar.close, None])
        preds = [m.predict(x) for m in self.models]
        # Weights use only losses on already matured labels.
        weights = [1 / max(m.mse, .1) for m in self.models]
        total = sum(weights)
        mu = sum(p[0]*w for p, w in zip(preds, weights)) / total
        disagree = math.sqrt(sum(w*(p[0]-mu)**2 for p, w in zip(preds, weights)) / total)
        se = math.sqrt(sum(w*p[1]**2 for p, w in zip(preds, weights)) / total)
        # Overlapping horizon labels are dependent: inflate uncertainty, do not call this a CI.
        uncertainty = (se * math.sqrt(self.horizon) + disagree) * scale
        contributions = {name: round(sum(w*m.w[i]*x[i] for w, m in zip(weights, self.models))
                                     / total * scale, 9) for i, name in enumerate(NAMES)}
        h = hashlib.sha256(json.dumps([m.w for m in self.models], separators=(",", ":")).encode()).hexdigest()[:16]
        return Forecast(mu*scale, uncertainty, mu*scale-self.penalty*uncertainty,
                        atr, self.models[0].samples, disagree*scale, contributions, h)
