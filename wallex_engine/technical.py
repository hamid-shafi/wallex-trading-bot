"""Closed six-hour indicators and deterministic, auditable entry filters."""
import math
from .data import aggregate


WINDOW = 100
INTERVAL = 21600


def indicators(candles):
    """Use exactly 100 complete candles so batch and portable inference agree."""
    b = candles[-WINDOW:]
    if len(b) != WINDOW or any(y.ts-x.ts != INTERVAL for x,y in zip(b,b[1:])):
        return {'technical_available':False}
    closes = [v.close for v in b]
    changes = [y-x for x,y in zip(closes,closes[1:])]
    gain = sum(max(0,v) for v in changes[:14])/14
    loss = sum(max(0,-v) for v in changes[:14])/14
    for change in changes[14:]:
        gain = (13*gain+max(0,change))/14
        loss = (13*loss+max(0,-change))/14
    rsi = 50. if gain+loss < 1e-15 else 100*gain/(gain+loss)
    fast = slow = closes[0]; signal = 0.; line = 0.
    for close in closes[1:]:
        fast += (close-fast)*2/13
        slow += (close-slow)*2/27
        line = fast-slow
        signal += (line-signal)*2/10
    last,prev = b[-1],b[-2]
    span = max(last.high-last.low,last.close*1e-12)
    body = abs(last.close-last.open)
    upper = last.high-max(last.open,last.close)
    lower = min(last.open,last.close)-last.low
    bearish_engulfing = (prev.close>prev.open and last.close<last.open
        and last.open>=prev.close and last.close<=prev.open)
    bullish_engulfing = (prev.close<prev.open and last.close>last.open
        and last.open<=prev.close and last.close>=prev.open)
    shooting_star = upper>=2*max(body,.05*span) and lower<=.2*span and body<=.35*span and (last.close-last.low)/span<=.4
    hammer = lower>=2*max(body,.05*span) and upper<=.2*span and body<=.35*span and (last.close-last.low)/span>=.6
    return {'technical_available':True,'technical_closed_at':last.ts+INTERVAL,
        'rsi_6h':rsi,'macd_6h':line/last.close,'macd_histogram_6h':(line-signal)/last.close,
        'body_ratio_6h':body/span,'upper_wick_ratio_6h':upper/span,'lower_wick_ratio_6h':lower/span,
        'bearish_engulfing_6h':bool(bearish_engulfing),'bullish_engulfing_6h':bool(bullish_engulfing),
        'shooting_star_6h':bool(shooting_star),'hammer_6h':bool(hammer),'doji_6h':body/span<=.1}


def context(history, when=None):
    if not history: return {'technical_available':False}
    when = history[-1].ts+3600 if when is None else when
    # Remove every candle whose close is still in the future, even if supplied.
    complete = [b for b in history if b.ts+3600<=when]
    buckets = aggregate(complete,INTERVAL,3600)
    return indicators([b for b in buckets if b.ts+INTERVAL<=when])


def apply_filters(spec, aux):
    """Entry vetoes only. Values returned here are directional utilities, not returns."""
    utility = aux['committee']
    parts = {'filter_reasons':[],'directional_utility_is_expected_profit':False}
    if utility<=0: return utility,parts
    rules = spec.get('filters',[])
    if 'quality' in rules and (aux['strength']<.45 or not aux['trend_entry']):
        parts['filter_reasons'].append('weak_or_mixed_trend')
    if 'cost_regime' in rules and not (aux['raw_momentum_168']>=.023 and aux['breadth_720']>=.6):
        parts['filter_reasons'].append('weak_momentum_relative_to_cost_proxy')
    if any(v in rules for v in ('macd_rsi','candles')):
        if not aux.get('technical_available',False):
            parts['filter_reasons'].append('technical_history_unavailable')
        else:
            if 'macd_rsi' in rules and not (45<=aux['rsi_6h']<=75 and aux['macd_6h']>0 and aux['macd_histogram_6h']>=0):
                parts['filter_reasons'].append('macd_or_rsi_veto')
            if 'candles' in rules and (aux['bearish_engulfing_6h'] or aux['shooting_star_6h']):
                parts['filter_reasons'].append('bearish_candle_veto')
    for key in ('rsi_6h','macd_6h','macd_histogram_6h','bearish_engulfing_6h','shooting_star_6h','technical_closed_at'):
        if key in aux: parts[key]=aux[key]
    return (0. if parts['filter_reasons'] else utility),parts
