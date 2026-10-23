"""Read-only operator check; never calls the live engine or sends orders."""
import json
from contextlib import closing
import os
import time
import sqlite3
from pathlib import Path
from dataclasses import asdict
from .config import Config
from .data import digest_file
from .engine import Portfolio,Intent
from .fees import fetch_fee_rates
from .operator_trial import ROUTES,RUN_DIR,operator_configuration
from .audit import source_hash
from .runtime import check_balances, open_order_list
from .session_limits import load_operator_limits
from .wallex import Wallex, APIError,validate_market


class CheckWallex(Wallex):
    PATHS=frozenset(('/v1/markets','/v1/account/balances',
                     '/v1/account/openOrders','/v1/account/fee'))

    def request(self,method,path,params=None,body=None,private=False):
        if method!='GET' or path not in self.PATHS or body is not None:
            raise APIError('operator check allows read-only routes only')
        return super().request(method,path,params,body,private)


def inspect(api,seed,run=None):
    seed=Path(seed)
    c=Config.load(seed/'config.json');c.routes=ROUTES.copy()
    c.validate();limits=load_operator_limits(seed,os.environ).validate(c.initial_cash)
    model=json.loads((seed/'model.json').read_text())
    markets=api.markets()
    if any(s not in markets for s in c.symbols):raise APIError('configured market absent')
    balances=api.account(c,'balances')
    row=balances.get('balances',balances).get('TMN',{})
    result=dict(initial_cash_tmn=c.initial_cash,initial_cash_irr=c.initial_cash*10,
                gross_cap=c.gross_cap,max_drawdown=c.max_drawdown,
                limits=asdict(limits),raw_balance_value=row.get('value'),
                raw_balance_asset=row.get('asset','TMN'),
                model_sha256=digest_file(seed/'model.json'),
                model_research_only=model.get('research_only'),
                real_orders_sent=0,checks={},ready_for_manual_activation=False)
    checks=result['checks']
    checks['session_deadline']='pass' if limits.ends_at is None or time.time()<limits.ends_at<=time.time()+604800 else 'blocked: deadline expired or too distant'
    portfolio=Portfolio(c.initial_cash)
    if run is not None and (Path(run)/'session/session.sqlite').exists():
        path=Path(run)/'session/session.sqlite'
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            state={k:json.loads(v) for k,v in db.execute('select key,value from state')}
            for intent,fill in db.execute("select intent,result from orders where status='DONE' order by seq"):
                intent,fill=Intent(**json.loads(intent)),json.loads(fill)
                if fill['quantity']:
                    portfolio.apply(intent,fill['quantity'],fill['price'],fill['fee'],fill['ts'])
                    portfolio.fees+=fill.get('base_fee_tmn',0)
            checks['no_ambiguous_bot_orders']='pass' if not db.execute("select count(*) from orders where status IS NOT 'DONE'").fetchone()[0] else 'blocked'
        identity=state.get('identity',{})
        checks['persisted_session_identity']='pass' if (identity.get('source')==source_hash() and
            identity.get('trial_limits')==asdict(limits) and identity.get('config')==asdict(operator_configuration(run,seed))) else 'blocked: offline plan migration required'
        checks['persisted_stop']='pass' if (not state.get('engine',{}).get('halted') and not state.get('trial',{}).get('reason') and
            not (Path(run)/'session/KILL').exists()) else 'blocked: recorded stop requires reconciliation'
        result['inventory_basis']='confirmed_bot_fills'
    try:
        for s in c.symbols:validate_market(markets[s])
        checks['execution_market_schema']='pass'
    except APIError:checks['execution_market_schema']='blocked: required market metadata missing/invalid'
    try:
        check_balances(api,c,portfolio,markets)
        checks['balance_units_and_inventory']='pass'
    except APIError:
        checks['balance_units_and_inventory']='blocked: reconcile balance units/inventory; no scaling inferred'
    checks['no_open_orders']='pass' if not open_order_list(api.account(c,'open_orders')) else 'blocked'
    try:
        result['account_fee_rates']=fetch_fee_rates(api,c)
        checks['fees']='pass'
    except APIError:
        checks['fees']='blocked: account fees unavailable, invalid, or above configured budget'
    checks['model_dates']='pass' if model['training_cutoff']<=time.time()<model['valid_until'] else 'blocked'
    checks['risk_config']='pass' if (c.initial_cash==5e6 and c.gross_cap==.8 and c.max_drawdown==.15) else 'blocked'
    result['ready_for_manual_activation']=all(v=='pass' for v in checks.values())
    return result


def main():
    result=inspect(CheckWallex(),os.getenv('SEED_DIR','/app/deploy/seed'),Path(os.getenv('DATA_DIR','/data'))/RUN_DIR)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['ready_for_manual_activation'] else 2


if __name__=='__main__':raise SystemExit(main())
