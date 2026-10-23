"""Read-only journal fee reporting from synthetic, local terminal results."""
import json
import sqlite3
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from observer.server import journal, order_fee_details
from wallex_engine.config import Config
from wallex_engine.engine import Intent


class ObserverFeeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        directory = self.root / 'synthetic-spot' / 'session'
        directory.mkdir(parents=True)
        self.path = directory / 'session.sqlite'
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT)')
            db.execute('CREATE TABLE orders (seq INTEGER, id TEXT, intent TEXT, status TEXT, result TEXT)')
            identity = {'mode': 'live', 'config': asdict(Config(symbols=['BTCTMN']))}
            db.execute('INSERT INTO state VALUES (?,?)', ('identity', json.dumps(identity)))

    def add_order(self, oid, side, quantity, result, status='DONE'):
        intent = asdict(Intent('BTCTMN', side, quantity, 'synthetic', .02))
        with sqlite3.connect(self.path) as db:
            sequence = db.execute('SELECT COUNT(*) FROM orders').fetchone()[0] + 1
            db.execute('INSERT INTO orders VALUES (?,?,?,?,?)',
                       (sequence, oid, json.dumps(intent), status,
                        json.dumps(result) if result is not None else None))

    def test_total_includes_base_fee_and_matches_ledger_without_double_charge(self):
        buy = {'quantity': 1.996, 'price': 200_000 / 1.996, 'fee': 0.,
               'base_fee_tmn': 400., 'base_fee_quantity': .004, 'quote_fee_tmn': 0., 'ts': 100}
        proceeds = 1.996 * 99_900
        sell = {'quantity': 1.996, 'price': 99_900., 'fee': proceeds * .002,
                'base_fee_tmn': 0., 'base_fee_quantity': 0.,
                'quote_fee_tmn': proceeds * .002, 'ts': 101}
        self.add_order('buy', 'BUY', 2., buy)
        self.add_order('sell', 'SELL', 1.996, sell)
        data = journal(self.root, selected_run='synthetic-spot')
        rows = {row['id']: row for row in data['orders']}
        self.assertEqual(rows['buy']['fee'], 0., 'legacy fee remains quote-only')
        self.assertEqual(rows['buy']['quote_fee_tmn'], 0.)
        self.assertEqual(rows['buy']['base_fee_tmn'], 400.)
        self.assertEqual(rows['buy']['base_fee_quantity'], .004)
        self.assertEqual(rows['buy']['fee_base_asset'], 'BTC')
        self.assertEqual(rows['buy']['fee_quote_asset'], 'TMN')
        self.assertEqual(rows['buy']['fee_tmn'], 400.)
        self.assertAlmostEqual(rows['sell']['fee_tmn'], proceeds * .002)
        self.assertEqual(rows['buy']['executed'], buy['quantity'])
        self.assertEqual(rows['buy']['price'], buy['price'])
        self.assertAlmostEqual(sum(row['fee_tmn'] for row in rows.values()), data['ledger']['fees_tmn'])
        expected_cash = 4_800_000 + proceeds * .998
        self.assertAlmostEqual(data['ledger']['cash_tmn'], expected_cash)
        self.assertAlmostEqual(data['ledger']['realized_pnl_tmn'], expected_cash - 5_000_000)

    def test_legacy_base_valuation_is_reported_without_inventing_asset_quantity(self):
        self.add_order('legacy-buy', 'BUY', 2., {
            'quantity': 1.996, 'price': 200_000 / 1.996, 'fee': 0.,
            'base_fee_tmn': 400., 'ts': 100})
        row = journal(self.root, selected_run='synthetic-spot')['orders'][0]
        self.assertEqual(row['fee_tmn'], 400.)
        self.assertIsNone(row['base_fee_quantity'])
        self.assertEqual(row['fee'], 0.)

    def test_confirmed_zero_fill_is_zero_but_pending_fee_remains_unknown(self):
        self.add_order('cancelled', 'BUY', 2., {'quantity': 0., 'price': 0., 'fee': 0., 'ts': 100})
        self.add_order('pending', 'BUY', 2., None, 'SENDING')
        data = journal(self.root, selected_run='synthetic-spot')
        rows = {row['id']: row for row in data['orders']}
        self.assertEqual(rows['cancelled']['fee_tmn'], 0.)
        for key in ('fee_tmn', 'quote_fee_tmn', 'base_fee_tmn', 'base_fee_quantity'):
            self.assertIsNone(rows['pending'][key], key)
        self.assertEqual(data['ledger']['fees_tmn'], 0.)

    def test_unknown_currency_or_invalid_fee_is_not_a_tmn_total(self):
        non_tmn = order_fee_details('BTCUSDT', 'DONE', {'fee': 2., 'base_fee_tmn': 3.})
        self.assertEqual(non_tmn['fee_quote_asset'], 'USDT')
        self.assertIsNone(non_tmn['fee_tmn'])
        for value in (True, -1, float('nan'), float('inf')):
            with self.subTest(value=value):
                self.assertIsNone(order_fee_details('BTCTMN', 'DONE', {'fee': value})['fee_tmn'])


if __name__ == '__main__':
    unittest.main()
