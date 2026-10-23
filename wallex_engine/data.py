from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import random


@dataclass(frozen=True)
class Bar:
    ts: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def validate(self):
        vals = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(x) for x in vals) or min(vals[:4]) <= 0 or self.volume < 0:
            raise ValueError("invalid OHLCV")
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("inconsistent OHLC")
        if self.ts <= 0:
            raise ValueError("invalid bar timestamp")
        return self


def save_bars(path, bars):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Bar.__dataclass_fields__))
        w.writeheader()
        for b in sorted(bars, key=lambda b: (b.ts, b.symbol)):
            w.writerow(asdict(b))


def load_bars(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        bars = [Bar(int(r["ts"]), r["symbol"], **{k: float(r[k]) for k in
                ("open", "high", "low", "close", "volume")}).validate() for r in csv.DictReader(f)]
    return bars


def frames(bars, symbols, interval):
    groups = defaultdict(dict)
    for b in bars:
        b.validate()
        if b.symbol not in symbols:
            continue
        if b.symbol in groups[b.ts]:
            raise ValueError(f"duplicate bar: {b.ts}/{b.symbol}")
        if b.ts % interval:
            raise ValueError("bars must be aligned to UTC interval")
        groups[b.ts][b.symbol] = b
    times = sorted(groups)
    if not times:
        raise ValueError("no data")
    for i, t in enumerate(times):
        if set(groups[t]) != set(symbols):
            raise ValueError(f"missing market at {t}; do not forward fill prices")
        if i and t - times[i - 1] != interval:
            raise ValueError(f"data gap at {t}; fetch/repair data before evaluation")
    return [groups[t] for t in times]


def aggregate(bars, interval, source_interval=60):
    """Never trust server resolution aliases: build complete 5m/15m bars from 1m."""
    if interval % source_interval:
        raise ValueError("invalid aggregation interval")
    groups = defaultdict(list)
    for b in bars:
        groups[(b.symbol, b.ts // interval * interval)].append(b)
    result = []
    for (symbol, ts), items in sorted(groups.items()):
        items.sort(key=lambda b: b.ts)
        if [b.ts for b in items] != list(range(ts, ts + interval, source_interval)):
            continue
        result.append(Bar(ts, symbol, items[0].open, max(b.high for b in items),
                          min(b.low for b in items), items[-1].close, sum(b.volume for b in items)))
    return result


def synthetic(symbols, count=2400, interval=300, seed=73):
    """Engineering fixture with regimes; NEVER evidence of real profitability."""
    rng = random.Random(seed)
    prices = {s: 100_000.0 * (i + 1) for i, s in enumerate(symbols)}
    start = 1_735_689_600 // interval * interval
    result = []
    for i in range(count):
        common = rng.gauss(0, .0025)
        regime = (i // 180) % 4
        drift = (.0014, -.0012, .0001, 0)[regime]
        for j, s in enumerate(symbols):
            o = prices[s]
            ret = drift + common * .7 + rng.gauss(0, .0015) + .0004 * math.sin(i / 13 + j)
            c = o * math.exp(ret)
            wiggle = abs(rng.gauss(0, .002))
            result.append(Bar(start + i * interval, s, o, max(o, c) * (1 + wiggle),
                              min(o, c) * (1 - wiggle), c, rng.uniform(100, 300)))
            prices[s] = c
    return result


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(path, source, extra=None):
    out = {"source": source, "sha256": digest_file(path), **(extra or {})}
    Path(str(path) + ".meta.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
