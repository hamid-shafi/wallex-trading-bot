"""Offline runtime/executor integration with a synthetic spot account only."""
import contextlib
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from wallex_engine.config import Config
from wallex_engine.data import Bar, save_bars
from wallex_engine.engine import Engine, Intent
from wallex_engine.execution import Journal
from wallex_engine.runtime import run_session
from wallex_engine.session_limits import SessionLimits
from wallex_engine.wallex import APIError


NOW = 1_792_780_200
SYMBOL = 'BTCTMN'


class FixtureConfig(Config):
    """Bypass arming only inside this test; no real client is constructed."""
    operator_spot = True

    def live_preflight(self, now):
        pass


class SyntheticSpot:
    def __init__(self, now, initial_cash):
        self.now = now
        self.cash = initial_cash
        self.btc = 0.
        self.calls = []
        self.submitted = []
        self.orders = {}
        self.fee_error = None
        self.fee_rate = '0.002'
        self.sell_fee_asset = 'TMN'
        self.max_qty = None

    def markets(self):
        return {SYMBOL: dict(baseAsset='BTC', quoteAsset='TMN', stepSize=8,
                             tickSize=2, minQty=1e-8, minNotional=100_000,
                             maxQty=self.max_qty, orderTypes=['LIMIT'])}

    def depth(self, symbol):
        assert symbol == SYMBOL
        self.calls.append(('depth', symbol))
        return dict(asks=[(100_000., 100.)], bids=[(99_900., 100.)],
                    received_at=self.now)

    def history(self, *args):
        raise AssertionError('fixture should use its existing closed candle')

    def request(self, method, path, params=None, body=None, private=False):
        assert (method, path, private) == ('GET', '/v1/account/fee', True)
        assert params is None and body is None
        self.calls.append(('fees',))
        if self.fee_error:
            raise self.fee_error
        return {'success': True, 'result': {SYMBOL: {
            'makerFeeRate': self.fee_rate, 'takerFeeRate': self.fee_rate}}}

    def account(self, config, name, method='GET', client_id=None, body=None):
        self.calls.append((method, name))
        if name == 'balances' and method == 'GET':
            return {'balances': {'TMN': {'value': self.cash, 'locked': 0},
                                 'BTC': {'value': self.btc, 'locked': 0}}}
        if name == 'open_orders' and method == 'GET':
            return []
        if name == 'order' and method == 'GET':
            return deepcopy(self.orders[client_id])
        if name != 'orders' or method != 'POST':
            raise AssertionError(f'unexpected synthetic account call: {method} {name}')
        payload = deepcopy(body)
        self.submitted.append(payload)
        quantity = float(payload['quantity'])
        side = payload['side']
        price = 100_000. if side == 'BUY' else 99_900.
        asset = 'BTC' if side == 'BUY' else self.sell_fee_asset
        fee = quantity * .002 if asset == 'BTC' else quantity * price * .002
        if side == 'BUY':
            self.cash -= quantity * price
            self.btc += quantity - fee
        else:
            self.cash += quantity * price - (fee if asset == 'TMN' else 0)
            self.btc -= quantity + (fee if asset == 'BTC' else 0)
        order = dict(clientOrderId=payload['client_id'], symbol=SYMBOL, side=side,
                     active=False, executedQty=quantity,
                     fills=[dict(quantity=quantity, price=price, fee=fee, feeAsset=asset)])
        self.orders[payload['client_id']] = order
        return {'clientOrderId': payload['client_id']}


class RuntimeFeeTests(unittest.TestCase):
    def setUp(self):
        # A mistaken transport fallback fails before any socket can be opened.
        self.addCleanup(patch.stopall)
        patch('socket.socket', side_effect=AssertionError('network forbidden in fixture')).start()
        patch('wallex_engine.wallex.Wallex.request',
              side_effect=AssertionError('real API forbidden in fixture')).start()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / 'bars.csv'
        save_bars(self.csv, [Bar(NOW // 3600 * 3600 - 3600, SYMBOL,
                                 100_000, 101_000, 99_000, 100_000, 100)])
        self.config = FixtureConfig(symbols=[SYMBOL], interval=3600, fee_bps=20)
        self.api = SyntheticSpot(NOW, self.config.initial_cash)
        self.limits = SessionLimits(None, None, 604_800, NOW + 60)

    def run_once(self, now, plan):
        self.api.now = now
        with contextlib.redirect_stdout(io.StringIO()), \
                patch('wallex_engine.runtime.time.time', return_value=now), \
                patch.object(Engine, 'plan', side_effect=plan):
            return run_session(self.api, self.config, self.csv, self.root / 'session',
                               mode='live', session_limits=self.limits)

    @staticmethod
    def buy_plan(*args):
        return [Intent(SYMBOL, 'BUY', 2., 'synthetic_entry', .02)], []

    @staticmethod
    def no_entry_plan(*args):
        raise AssertionError('entry planning must not run after the deadline')

    def test_base_fee_buy_then_full_inventory_quote_fee_exit_and_restart(self):
        first = self.run_once(NOW, self.buy_plan)
        self.assertEqual(first['fills'], 1)
        self.assertAlmostEqual(first['cash_tmn'], 4_800_000.)
        self.assertAlmostEqual(first['fees_tmn'], 400.)
        self.assertAlmostEqual(first['positions'][SYMBOL]['quantity'], 1.996)
        # Fee validation must precede the fresh submission book and the POST.
        fee_call = self.api.calls.index(('fees',))
        self.assertEqual(self.api.calls[fee_call + 1], ('depth', SYMBOL))
        self.assertLess(fee_call, self.api.calls.index(('POST', 'orders')))

        # Exit must still work when account fee lookup is unavailable.
        self.api.fee_error = APIError('synthetic fee endpoint unavailable')
        final = self.run_once(NOW + 61, self.no_entry_plan)
        sell = self.api.submitted[1]
        self.assertEqual(sell['side'], 'SELL')
        self.assertAlmostEqual(float(sell['quantity']), 1.996)
        proceeds = 1.996 * 99_900.
        expected_cash = 4_800_000. + proceeds * .998
        expected_fees = 400. + proceeds * .002
        self.assertAlmostEqual(final['cash_tmn'], expected_cash)
        self.assertAlmostEqual(final['realized_pnl_tmn'], expected_cash - 5_000_000.)
        self.assertAlmostEqual(final['fees_tmn'], expected_fees)
        self.assertEqual(final['positions'], {})
        self.assertEqual(final['fills'], 2)
        self.assertTrue(final['settled_to_tmn'])
        self.assertTrue(final['stopped'])
        self.assertEqual(final['trial']['reason'], 'time_limit')
        self.assertEqual(self.api.calls.count(('fees',)), 1)

        replayed = self.run_once(NOW + 62, self.no_entry_plan)
        for key in ('cash_tmn', 'realized_pnl_tmn', 'fees_tmn', 'fills', 'positions'):
            self.assertEqual(replayed[key], final[key], key)
        self.assertEqual(len(self.api.submitted), 2)
        self.assertEqual(self.api.calls.count(('fees',)), 1)
        self.assertAlmostEqual(self.api.btc, 0.)
        self.assertAlmostEqual(self.api.cash, final['cash_tmn'])

    def test_exchange_capped_sell_can_charge_base_fee_within_full_inventory(self):
        self.run_once(NOW, self.buy_plan)
        self.api.max_qty = 1.25
        self.api.sell_fee_asset = 'BTC'
        self.api.fee_error = APIError('synthetic fee endpoint unavailable')
        final = self.run_once(NOW + 61, self.no_entry_plan)
        self.assertEqual(float(self.api.submitted[-1]['quantity']), 1.25)
        self.assertAlmostEqual(final['positions'][SYMBOL]['quantity'], .7435)
        self.assertAlmostEqual(final['cash_tmn'], 4_924_875.)
        self.assertAlmostEqual(final['fees_tmn'], 400. + .0025 * 99_900.)
        self.assertFalse(final['settled_to_tmn'])
        self.assertEqual(final['fills'], 2)
        self.assertEqual(self.api.calls.count(('fees',)), 1)
        self.assertAlmostEqual(self.api.btc, final['positions'][SYMBOL]['quantity'])

    def test_rejected_fee_lookup_blocks_buy_before_fresh_book_or_submission(self):
        for case in ('above_cap', 'unavailable'):
            with self.subTest(case=case):
                self.api = SyntheticSpot(NOW, self.config.initial_cash)
                if case == 'above_cap':
                    self.api.fee_rate = '0.0021'
                else:
                    self.api.fee_error = APIError('synthetic fee endpoint unavailable')
                # Independent journals ensure each case attempts a fresh BUY decision.
                with tempfile.TemporaryDirectory() as folder:
                    previous = self.root
                    self.root = Path(folder)
                    try:
                        with self.assertRaises(APIError):
                            self.run_once(NOW, self.buy_plan)
                        self.assertEqual(self.api.submitted, [])
                        self.assertEqual(self.api.calls.count(('depth', SYMBOL)), 1)
                        self.assertEqual(self.api.calls.count(('fees',)), 1)
                        journal = Journal(self.root / 'session' / 'session.sqlite')
                        try:
                            self.assertEqual(journal.pending(), [])
                            self.assertEqual(journal.db.execute('SELECT COUNT(*) FROM orders').fetchone()[0], 0)
                        finally:
                            journal.close()
                    finally:
                        self.root = previous


if __name__ == '__main__':
    unittest.main()
