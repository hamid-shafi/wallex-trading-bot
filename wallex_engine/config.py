from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math


@dataclass
class Config:
    symbols: list[str] = field(default_factory=lambda: ["BTCTMN", "ETHTMN"])
    initial_cash: float = 5_000_000.0
    interval: int = 300
    horizon: int = 12
    min_samples: int = 300
    fee_bps: float = 20.0
    spread_bps: float = 12.0
    slippage_bps: float = 8.0
    edge_bps: float = 12.0
    uncertainty_penalty: float = 0.6
    risk_fraction: float = 0.003
    position_cap: float = 0.25
    gross_cap: float = 0.50
    portfolio_stop_risk: float = 0.008
    max_positions: int = 2
    max_drawdown: float = 0.05
    daily_loss: float = 0.02
    min_notional: float = 100_000.0
    dust_cleanup_fraction: float = 0.03
    participation: float = 0.01
    stop_atr: float = 2.5
    min_stop_fraction: float = 0.007
    cooldown: int = 3
    max_spread_bps: float = 35.0
    max_book_slippage_bps: float = 20.0
    stale_seconds: int = 20
    settlement_buffer: int = 900
    challenge_end: int = 0
    allowed_markets_verified: bool = False
    account_binding_verified: bool = False
    fee_verified: bool = False
    live_enabled: bool = False
    model_artifact: str = ""
    strategy: str = "forecast"
    trend_fast: int = 24
    trend_slow: int = 168
    trend_entry_bps: float = 25.
    trend_exit_bps: float = 25.
    max_holding_bars: int = 0
    entry_buffer_bars: int = 0
    trailing_atr: float = 0.
    meta_min_probability: float = .5
    news_path: str = ""
    news_required: bool = False
    paper_run: str = "runs/agent-paper"
    # Challenge credit wallet must be explicitly routed; main spot is NOT a fallback.
    routes: dict = field(default_factory=dict)

    def validate(self):
        for name in ("interval","horizon","min_samples","max_positions","cooldown","stale_seconds",
                     "settlement_buffer","challenge_end","trend_fast","trend_slow","max_holding_bars","entry_buffer_bars"):
            v = getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int):
                raise ValueError(f"{name} must be an integer")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be nonempty and unique")
        if any(not s.isalnum() or not s.endswith("TMN") for s in self.symbols):
            raise ValueError("this engine settles only direct TMN markets")
        if self.interval not in (60, 300, 900, 3600):
            raise ValueError("interval must be 60,300,900,3600 seconds")
        if not (1 <= self.horizon <= 96 and self.min_samples >= 50):
            raise ValueError("invalid horizon/min_samples")
        if not (0 < self.position_cap <= self.gross_cap <= 0.8):
            raise ValueError("invalid exposure limits")
        if not (0 < self.risk_fraction <= self.portfolio_stop_risk <= 0.03):
            raise ValueError("invalid risk budget")
        if not (0 < self.daily_loss <= self.max_drawdown <= 0.15):
            raise ValueError("invalid loss limits")
        for k, v in asdict(self).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not math.isfinite(v) or v < 0:
                    raise ValueError(f"invalid numeric setting: {k}")
        if self.initial_cash <= 0 or self.max_positions < 1 or self.min_notional <= 0:
            raise ValueError("invalid cash/positions/notional")
        if not 0 < self.participation <= 0.1:
            raise ValueError("participation must be in (0,0.1]")
        if not 0<=self.dust_cleanup_fraction<=0.05:
            raise ValueError("dust cleanup budget must be <=5% of equity")
        if self.strategy not in ("forecast","trend_meta","trend"):
            raise ValueError("unsupported strategy")
        if not 2<=self.trend_fast<self.trend_slow<=720 or not 0<=self.meta_min_probability<=1:
            raise ValueError("invalid trend/meta settings")
        return self

    @classmethod
    def load(cls, path=None):
        return cls(**(json.loads(Path(path).read_text(encoding="utf-8")) if path else {})).validate()

    def round_trip_cost(self):
        return (2 * self.fee_bps + self.spread_bps + 2 * self.slippage_bps) / 10_000

    def live_preflight(self, now):
        self.validate()
        if self.model_artifact:
            model=json.loads(Path(self.model_artifact).read_text(encoding="utf-8"))
            if model.get("research_only",True):
                raise ValueError("trained artifact is restricted to backtesting and shadow; live trading is blocked")
        flags = (self.live_enabled, self.allowed_markets_verified,
                 self.account_binding_verified, self.fee_verified)
        if not all(flags):
            raise ValueError("live disabled: verify challenge markets, credit account routes and fees")
        if self.challenge_end <= now + self.settlement_buffer:
            raise ValueError("challenge end missing or already in settlement window")
        for name in ("balances", "open_orders", "orders", "order", "cancel"):
            if not self.routes.get(name, "").startswith("/v"):
                raise ValueError(f"verified account route required: {name}")
