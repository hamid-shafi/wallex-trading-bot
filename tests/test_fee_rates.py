"""Fee-rate tests use local fixtures; they never contact an exchange."""
import unittest
from unittest.mock import patch

from wallex_engine.config import Config
from wallex_engine.fees import fetch_fee_rates, validate_fee_rates
from wallex_engine.operator_trial import SpotWallex
from wallex_engine.wallex import APIError, Wallex


class FeeRateTests(unittest.TestCase):
    def test_fractional_rates_and_both_liquidity_roles(self):
        rates = validate_fee_rates({'BTCTMN': {
            'makerFeeRate': '0.00200000', 'takerFeeRate': '0.00350000',
            'recent_days_sum': '1000000',
        }}, ['BTCTMN'], 35)
        self.assertEqual(rates['BTCTMN'], {'makerFeeRate': .002, 'takerFeeRate': .0035})

    def test_exact_boundary_is_accepted_but_either_rate_above_is_rejected(self):
        for key in ('makerFeeRate', 'takerFeeRate'):
            with self.subTest(key=key):
                row = {'makerFeeRate': '0', 'takerFeeRate': '0'}
                row[key] = '0.003500000000000001'
                with self.assertRaises(APIError):
                    validate_fee_rates({'BTCTMN': row}, ['BTCTMN'], 35)

    def test_malformed_or_missing_fees_never_pass(self):
        valid = {'makerFeeRate': '0.003', 'takerFeeRate': '0.003'}
        for raw in (None, [], {}, {'BTCTMN': {}}, {'BTCTMN': []}):
            with self.subTest(raw=raw), self.assertRaises(APIError):
                validate_fee_rates(raw, ['BTCTMN'], 35)
        for invalid in (True, False, None, '', 'NaN', 'sNaN', 'Infinity', '-0.001', '0.35', 35):
            with self.subTest(rate=invalid), self.assertRaises(APIError):
                validate_fee_rates({'BTCTMN': {**valid, 'makerFeeRate': invalid}}, ['BTCTMN'], 35)
        with self.assertRaises(APIError):
            validate_fee_rates({'BTCTMN': valid}, ['BTCTMN', 'ETHTMN'], 35)

    def test_missing_or_failed_response_is_an_api_error(self):
        class Fake:
            def request(self, *args, **kwargs):
                return self.response
        api = Fake()
        for response in (None, [], {}, {'success': False, 'result': {}}, {'result': None}):
            api.response = response
            with self.subTest(response=response), self.assertRaises(APIError):
                fetch_fee_rates(api, Config())

    def test_operator_allows_only_reading_account_fees(self):
        client = SpotWallex(key='synthetic-offline-fixture')
        with patch.object(Wallex, 'request', return_value={'result': {}}) as transport:
            self.assertEqual(client.request('GET', '/v1/account/fee', private=True), {'result': {}})
            self.assertEqual(transport.call_count, 1)
            for method in ('POST', 'DELETE'):
                with self.assertRaises(APIError):
                    client.request(method, '/v1/account/fee', private=True)
            self.assertEqual(transport.call_count, 1)


if __name__ == '__main__':
    unittest.main()
