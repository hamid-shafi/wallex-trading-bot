"""Validate Wallex's fractional maker/taker rates without guessing units.

The account fee endpoint reports fractions: 0.0035 means 35 basis points.
Actual fills remain the authority for both fee amounts and fee assets.
"""
from decimal import Decimal, InvalidOperation

from .wallex import APIError


def validate_fee_rates(result, symbols, max_bps):
    """Return verified rates or reject missing, malformed, or excessive fees."""
    try:
        if isinstance(max_bps, bool):
            raise ValueError('boolean fee cap')
        cap = Decimal(str(max_bps)) / Decimal(10_000)
        if not cap.is_finite() or not 0 <= cap < 1 or not isinstance(result, dict):
            raise ValueError('invalid fee cap or response')
        verified = {}
        for symbol in symbols:
            row = result[symbol]
            if not isinstance(row, dict):
                raise ValueError('invalid market fee row')
            rates = {}
            for key in ('makerFeeRate', 'takerFeeRate'):
                raw = row[key]
                if isinstance(raw, bool):
                    raise ValueError('boolean fee rate')
                rate = Decimal(str(raw))
                if not rate.is_finite() or not 0 <= rate <= cap:
                    raise ValueError('fee outside configured budget')
                rates[key] = float(rate)
            verified[symbol] = rates
        return verified
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise APIError('account fee rates missing, invalid, or above configured budget') from exc


def fetch_fee_rates(api, config):
    response = api.request('GET', '/v1/account/fee', private=True)
    if not isinstance(response, dict) or response.get('success') is False:
        raise APIError('invalid account fee response')
    return validate_fee_rates(response.get('result'), config.symbols, config.fee_bps)
