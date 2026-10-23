"""Causal price/volume risk features shared by replay and portable inference."""
import math
import statistics


def context(histories,symbol):
    bars=histories[symbol]; latest=bars[-1].ts
    b=bars[-169:]
    if len(b)<169 or any(y.ts-x.ts!=3600 for x,y in zip(b,b[1:])):
        return None
    returns=[math.log(y.close/x.close) for x,y in zip(b,b[1:])]
    volratio=statistics.pstdev(returns[-24:])/max(.0003,statistics.pstdev(returns))
    visible=[h[-168:] for h in histories.values() if len(h)>=168 and h[-1].ts==latest
             and all(y.ts-x.ts==3600 for x,y in zip(h[-168:-1],h[-167:]))]
    breadth=sum(h[-1].close>statistics.fmean(b.close for b in h) for h in visible)/len(histories)
    liquid=statistics.median(b.close*b.volume for b in bars[-24:])*.01
    # Directional movement divided by total movement: a fixed, causal noise filter.
    efficiency=abs(sum(returns))/max(1e-12,sum(abs(r) for r in returns))
    return {'volatility_ratio':volratio,'market_breadth':breadth,'median_participation_tmn':liquid,
            'trend_efficiency_168':efficiency}


def adjust(utility,atr,features,values,min_notional=100000.):
    if values is None:
        return min(0,utility),atr
    if 'volatility' in features:
        atr*=max(1.,min(2.,values['volatility_ratio']))
    if 'breadth' in features and values['market_breadth']<.6:
        utility=min(0,utility)
    if 'liquidity' in features and values['median_participation_tmn']<2*min_notional:
        utility=min(0,utility)
    if 'efficiency' in features and values['trend_efficiency_168']<.2:
        utility=min(0,utility)
    return utility,atr
