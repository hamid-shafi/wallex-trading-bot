"""Deterministic PAPER-only entry reductions; no execution or news text analysis.

Thresholds are conservative engineering heuristics, not fitted return predictions.
Closed candles, local book snapshots and locally observed news metadata are used.
"""
from dataclasses import replace
import math
from pathlib import Path
import statistics

from .gpr import GPR_NAMES
from .news import FEEDS, NewsIndex

POLICY = 'paper_risk_reduction_v1'
CAPS = ('position_cap', 'gross_cap', 'risk_fraction', 'portfolio_stop_risk')
FRESH_NEWS_SECONDS = 6 * 3600
STALE_NEWS_SECONDS = 12 * 3600


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def reduced_config(config, multiplier):
    if not finite(multiplier) or not 0 <= multiplier <= 1:
        raise ValueError('adaptive multiplier must be in [0,1]')
    return replace(config, **{name: getattr(config, name) * multiplier for name in CAPS})


def reduce_entry(intent, multiplier):
    """Never change an exit, including when all new entries are blocked."""
    if not finite(multiplier) or not 0 <= multiplier <= 1:
        raise ValueError('adaptive multiplier must be in [0,1]')
    if intent.side != 'BUY':
        return intent
    return None if multiplier == 0 else replace(intent, quantity=intent.quantity * multiplier)


def _market(config, history, book, now):
    factors, reasons = [1.], []
    info = {'multiplier': 0., 'daily_volatility': None, 'spread_bps': None,
            'depth_tmn': None, 'reasons': reasons}
    try:
        # Discard unclosed/future candles before computing any statistic.
        bars = [b for b in history if finite(b.ts) and b.ts + config.interval <= now][-25:]
        if (len(bars) < 25 or any(b.ts - a.ts != config.interval for a, b in zip(bars, bars[1:]))
                or any(not finite(b.close) or b.close <= 0 for b in bars)):
            reasons.append('volatility_unknown'); return info
        if now - (bars[-1].ts + config.interval) > config.interval + config.stale_seconds:
            reasons.append('candles_stale'); return info
        returns = [math.log(b.close / a.close) for a, b in zip(bars, bars[1:])]
        volatility = statistics.pstdev(returns) * math.sqrt(86400 / config.interval)
        if not finite(volatility):
            reasons.append('market_data_unknown'); return info
        info['daily_volatility'] = volatility
        vol_factor = (0. if volatility >= .20 else .25 if volatility >= .10 else
                      .5 if volatility >= .06 else .75 if volatility >= .03 else 1.)
        factors.append(vol_factor)
        if vol_factor < 1: reasons.append('volatility_extreme' if not vol_factor else 'volatility_elevated')
        received = book['received_at']
        if not finite(received) or not 0 <= now - received <= config.stale_seconds:
            reasons.append('book_stale'); return info
        asks, bids = book['asks'], book['bids']
        if not asks or not bids or any(not finite(p) or not finite(q) or p <= 0 or q <= 0
                                       for p, q in asks + bids):
            reasons.append('book_invalid'); return info
        ask, bid = min(p for p, q in asks), max(p for p, q in bids)
        if bid >= ask:
            reasons.append('book_invalid'); return info
        spread = (ask - bid) / (ask / 2 + bid / 2) * 10000
        if not finite(spread):
            reasons.append('book_invalid'); return info
        info['spread_bps'] = spread
        ratio = spread / config.max_spread_bps if config.max_spread_bps else math.inf
        spread_factor = 0. if ratio >= 1 else .5 if ratio >= .75 else .75 if ratio >= .5 else 1.
        factors.append(spread_factor)
        if spread_factor < 1: reasons.append('spread_blocked' if not spread_factor else 'spread_elevated')
        slip = config.max_book_slippage_bps / 10000
        depth = min(sum(p * q for p, q in asks if p <= ask * (1 + slip)),
                    sum(p * q for p, q in bids if p >= bid * (1 - slip)))
        if not finite(depth):
            reasons.append('book_invalid'); return info
        info['depth_tmn'] = depth
        ratio = depth / config.min_notional
        depth_factor = 0. if ratio < 1 else .25 if ratio < 2 else .5 if ratio < 5 else .75 if ratio < 10 else 1.
        factors.append(depth_factor)
        if depth_factor < 1: reasons.append('liquidity_blocked' if not depth_factor else 'liquidity_thin')
        info['multiplier'] = min(factors)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        reasons.append('market_data_unknown')
    return info


class AdaptiveRisk:
    def __init__(self, news_path):
        self.news_path = Path(news_path) if news_path else None
        self.index = None

    def _news(self, config, now):
        info = {'coverage': False, 'freshness_seconds': None, 'source_count': 0,
                'conflict_share': None, 'model_news_coverage': False,
                'signal_kind': 'keyword_metadata_not_verified_event_facts'}
        try:
            if self.news_path is None or not self.news_path.is_file():
                return 0., ['news_missing'], info
            if self.index is None:
                self.index = NewsIndex(self.news_path)
            self.index.refresh()
            # Use NewsIndex's maintained rows, but independently enforce local
            # first-seen causality for ALL row types, including imported articles.
            visible = []
            for row in self.index.rows:
                if not isinstance(row, dict):
                    raise ValueError('invalid news row')
                if (row.get('reconstructed') is True or row.get('historical_proxy') is True
                        or row.get('availability_kind') == 'reconstructed_provider_time_proxy'):
                    continue
                times = [row.get('available_at'), row.get('first_seen_at')]
                if row.get('kind') == 'article': times.append(row.get('published_at'))
                if not all(finite(t) and t > 0 for t in times):
                    raise ValueError('invalid news timestamp')
                if max(times) <= now:
                    visible.append(row)
            articles = {}
            for row in visible:
                if row.get('kind') != 'article' or not 0 <= now - row['published_at'] <= 86400:
                    continue
                if row.get('source') not in FEEDS:
                    continue
                topics = row.get('topics', {})
                flags = [topics.get(k) for k in ('conflict', 'crypto_security')]
                if any(not finite(v) or v not in (0, 1) for v in flags):
                    raise ValueError('invalid topic metadata')
                key = row.get('duplicate_key')
                if not isinstance(key, str) or not key:
                    raise ValueError('invalid duplicate identity')
                # Deterministic deduplication cannot increase the count by repeating a row.
                if key not in articles or (row['published_at'], row['source']) < (
                        articles[key]['published_at'], articles[key]['source']):
                    articles[key] = row
            articles = list(articles.values())
            latest = max((r['published_at'] for r in articles), default=None)
            age = now - latest if latest is not None else None
            sources = {r['source'] for r in articles if now - r['published_at'] <= FRESH_NEWS_SECONDS}
            info.update(freshness_seconds=age, source_count=len(sources), coverage=len(sources) >= 2)
            factors, reasons = [1.], []
            if not articles:
                factors.append(0.); reasons.append('news_unknown')
            elif age > STALE_NEWS_SECONDS:
                factors.append(0.); reasons.append('news_stale')
            elif age > FRESH_NEWS_SECONDS:
                factors.append(.25); reasons.append('news_aging')
            elif len(sources) < 2:
                factors.append(.5); reasons.append('news_coverage_limited')
            if articles:
                share = sum(r['topics']['conflict'] for r in articles) / len(articles)
                info['conflict_share'] = share
                conflict = (0. if share >= .7 and len(articles) >= 3 else
                            .5 if share >= .35 else .75 if share > 0 else 1.)
                factors.append(conflict)
                if conflict < 1:
                    reasons.append('conflict_keywords_strong' if not conflict else 'conflict_keywords_elevated')
                if sum(r['topics']['crypto_security'] for r in articles) >= 2:
                    factors.append(.25); reasons.append('security_keywords_elevated')
            vintages = []
            for row in visible:
                if row.get('kind') != 'gpr_vintage': continue
                vector = row.get('features')
                observation, vintage = row.get('latest_observation'), row.get('vintage_date')
                if (row.get('feature_names') != GPR_NAMES or not isinstance(vector, list) or len(vector) != len(GPR_NAMES)
                        or not all(finite(v) for v in vector) or vector[-1] != 1
                        or not finite(observation) or not finite(vintage)):
                    raise ValueError('invalid GPR metadata')
                if (0 < observation <= vintage <= row['available_at'] <= now
                        and now - row['available_at'] <= 15 * 86400
                        and now - observation <= 18 * 86400):
                    vintages.append(row)
            info['model_news_coverage'] = bool(vintages)
            if config.news_required and not vintages:
                factors.append(0.); reasons.append('model_news_unknown')
            if vintages:
                vector = max(vintages, key=lambda r: r['available_at'])['features']
                gpr_factor = .5 if vector[2] >= 2 or vector[3] >= .5 else .75 if vector[2] >= 1 or vector[3] >= .2 else 1.
                factors.append(gpr_factor)
                if gpr_factor < 1: reasons.append('geopolitical_index_elevated')
            return min(factors), reasons, info
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            return 0., ['news_invalid_or_unavailable'], info

    def evaluate(self, config, histories, books, now):
        if not finite(now) or now <= 0:
            raise ValueError('invalid adaptive evaluation time')
        news_factor, reasons, news = self._news(config, now)
        markets = {s: _market(config, histories.get(s, []), books.get(s, {}), now) for s in config.symbols}
        multiplier = min([news_factor] + [m['multiplier'] for m in markets.values()])
        for symbol, market in markets.items():
            reasons.extend(f'{symbol}:{reason}' for reason in market['reasons'])
        return {'policy': POLICY, 'as_of': now, 'multiplier': multiplier,
                'reasons': sorted(set(reasons)), 'news': news, 'markets': markets,
                'effective_caps': {name: getattr(config, name) * multiplier for name in CAPS}}
