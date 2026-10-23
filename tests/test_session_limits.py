import json
from pathlib import Path
import tempfile
import unittest

from wallex_engine.execution import Journal
from wallex_engine.session_limits import (
    MAX_SESSION_SECONDS, SessionLimits, load_operator_limits,
)


class SessionLimitsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'session.sqlite'
        self.journal = Journal(self.path)
        self.addCleanup(lambda: self.journal.close())

    def restart(self):
        self.journal.close()
        self.journal = Journal(self.path)

    def test_optional_bands_do_not_stop_at_old_capital_values(self):
        limits = SessionLimits(floor_tmn=None, ceiling_tmn=None).validate(5_000_000)
        for equity in (3_500_000, 4_000_000, 6_000_000, 7_000_000):
            with self.subTest(equity=equity):
                self.assertIsNone(limits.check(self.journal, equity, 1_000)['reason'])

    def test_one_sided_bounds_are_inclusive(self):
        for limits, safe, trigger, expected in (
            (SessionLimits(floor_tmn=4_000_000, ceiling_tmn=None),
             4_000_001, 4_000_000, 'capital_floor'),
            (SessionLimits(floor_tmn=None, ceiling_tmn=6_000_000),
             5_999_999, 6_000_000, 'capital_ceiling'),
        ):
            with self.subTest(expected=expected):
                self.journal.db.execute('DELETE FROM state')
                self.journal.db.commit()
                limits.validate(5_000_000)
                self.assertIsNone(limits.check(self.journal, safe, 1_000)['reason'])
                self.assertEqual(limits.check(self.journal, trigger, 1_001)['reason'], expected)

    def test_invalid_capital_types_fail_as_value_errors(self):
        invalid = (True, False, float('nan'), float('inf'), -float('inf'),
                   '5000000', [], {}, 0, -1, 10**1000)
        for value in invalid:
            for field in ('initial', 'floor_tmn', 'ceiling_tmn'):
                with self.subTest(field=field, value=repr(value)):
                    kwargs = {'floor_tmn': None, 'ceiling_tmn': None}
                    initial = value if field == 'initial' else 5_000_000
                    if field != 'initial':
                        kwargs[field] = value
                    with self.assertRaises(ValueError):
                        SessionLimits(**kwargs).validate(initial)
        for limits in (SessionLimits(floor_tmn=5_000_000, ceiling_tmn=None),
                       SessionLimits(floor_tmn=None, ceiling_tmn=5_000_000),
                       SessionLimits(floor_tmn=6_000_000, ceiling_tmn=4_000_000)):
            with self.assertRaises(ValueError):
                limits.validate(5_000_000)

    def test_invalid_liquidation_values_cannot_start_a_trial(self):
        limits = SessionLimits(floor_tmn=None, ceiling_tmn=None)
        for value in (True, '5000000', None, float('nan'), float('inf'), [], 10**1000):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                limits.check(self.journal, value, 1_000)
        self.assertIsNone(self.journal.get('trial'))

    def test_absolute_deadline_persists_and_expires_after_restart(self):
        limits = SessionLimits(floor_tmn=None, ceiling_tmn=None, ends_at=4_000)
        started = limits.start(self.journal, 1_000)
        self.restart()
        before = limits.check(self.journal, 5_000_000, 3_999)
        self.assertEqual(before, started)
        self.assertEqual(limits.check(self.journal, 5_000_000, 4_000)['reason'], 'time_limit')
        self.restart()
        stopped = limits.check(self.journal, 5_000_000, 9_999, manual=True)
        self.assertEqual(stopped['deadline'], 4_000)
        self.assertEqual(stopped['started_at'], 1_000)
        self.assertEqual(stopped['triggered_at'], 4_000)
        self.assertEqual(stopped['reason'], 'time_limit')

    def test_relative_deadline_is_not_extended_by_restart(self):
        limits = SessionLimits(floor_tmn=None, ceiling_tmn=None, seconds=120)
        limits.start(self.journal, 1_000)
        self.restart()
        self.assertEqual(limits.start(self.journal, 1_100)['deadline'], 1_120)
        self.assertEqual(limits.check(self.journal, 5_000_000, 1_121)['reason'], 'time_limit')

    def test_new_deadlines_must_be_future_and_within_seven_days(self):
        for deadline in (999, 1_000, 1_001 + MAX_SESSION_SECONDS):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                SessionLimits(ends_at=deadline).start(self.journal, 1_000)
        self.assertIsNone(self.journal.get('trial'))
        self.assertEqual(SessionLimits(ends_at=1_000 + MAX_SESSION_SECONDS)
                         .start(self.journal, 1_000)['deadline'], 1_000 + MAX_SESSION_SECONDS)

    def test_changed_limits_cannot_silently_renew_a_trial(self):
        limits = SessionLimits(floor_tmn=None, ceiling_tmn=None, ends_at=2_000)
        original = limits.start(self.journal, 1_000)
        for changed in (SessionLimits(floor_tmn=None, ceiling_tmn=None, ends_at=3_000),
                        SessionLimits(ends_at=2_000)):
            with self.assertRaises(ValueError):
                changed.start(self.journal, 1_100)
        self.assertEqual(self.journal.get('trial'), original)

    def test_malformed_persisted_deadline_is_not_silently_repaired(self):
        limits = SessionLimits(floor_tmn=None, ceiling_tmn=None, ends_at=2_000)
        original = limits.start(self.journal, 1_000)
        for deadline in (True, '2000', 3_000):
            with self.subTest(deadline=deadline):
                self.journal.put('trial', {**original, 'deadline': deadline})
                with self.assertRaises(ValueError):
                    limits.check(self.journal, 5_000_000, 1_100)

    def test_malformed_persisted_reason_cannot_rearm_a_trial(self):
        limits = SessionLimits(floor_tmn=None, ceiling_tmn=None, ends_at=2_000)
        original = limits.start(self.journal, 1_000)
        for reason in ('', 'unknown', False, [], {}):
            with self.subTest(reason=reason):
                self.journal.put('trial', {**original, 'reason': reason})
                with self.assertRaises(ValueError):
                    limits.check(self.journal, 5_000_000, 1_100)

    def test_independent_stops_remain_latched_without_capital_bands(self):
        for flags, expected in (({'manual': True}, 'operator_stop'),
                                ({'halted': True}, 'drawdown_halt')):
            with self.subTest(expected=expected):
                self.journal.db.execute('DELETE FROM state')
                self.journal.db.commit()
                limits = SessionLimits(floor_tmn=None, ceiling_tmn=None)
                stopped = limits.check(self.journal, 5_000_000, 1_000, **flags)
                self.restart()
                self.assertEqual(stopped['reason'], expected)
                self.assertEqual(limits.check(self.journal, 7_000_000, 9_999), stopped)

    def test_existing_stop_precedence_is_unchanged(self):
        limits = SessionLimits(ends_at=1_100)
        limits.start(self.journal, 1_000)
        stopped = limits.check(self.journal, 3_000_000, 1_100, manual=True, halted=True)
        self.assertEqual(stopped['reason'], 'operator_stop')


class OperatorLimitLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.seed = Path(self.directory.name)
        self.plan = self.seed / 'session-plan.json'

    def write_plan(self, **overrides):
        plan = {'floor_tmn': None, 'ceiling_tmn': None,
                'ends_at': '2026-10-26T23:59:59+03:30'}
        plan.update(overrides)
        self.plan.write_text(json.dumps(plan), encoding='utf8')

    def test_monday_tehran_end_maps_to_exact_utc_second(self):
        self.write_plan()
        limits = load_operator_limits(self.seed, {}).validate(5_000_000)
        self.assertEqual(limits.ends_at, 1793046599)
        self.assertIsNone(limits.floor_tmn)
        self.assertIsNone(limits.ceiling_tmn)
        self.assertEqual(limits.seconds, MAX_SESSION_SECONDS)

    def test_explicit_environment_deadline_is_honored_with_or_without_plan(self):
        env = {'OPERATOR_END_AT': '2026-10-26T20:29:59+00:00'}
        without = load_operator_limits(self.seed, env)
        self.write_plan(ends_at='2026-10-25T23:59:59+03:30')
        with_plan = load_operator_limits(self.seed, env)
        self.assertEqual(without.ends_at, with_plan.ends_at)
        self.assertEqual(with_plan.ends_at, 1793046599)

    def test_invalid_deadline_never_falls_back_to_relative_session(self):
        for value in ('', False, 1, '2026-10-26T23:59:59',
                      '2026-10-26T23:59:59.500+03:30', 'not-a-date'):
            with self.subTest(value=value):
                self.write_plan(ends_at=value)
                with self.assertRaises(ValueError):
                    load_operator_limits(self.seed, {})

    def test_invalid_plan_and_values_are_rejected(self):
        for value in ([], {'floor_tmn': None},
                      {'floor_tmn': True, 'ceiling_tmn': None},
                      {'floor_tmn': None, 'ceiling_tmn': float('inf')}):
            with self.subTest(value=value):
                self.plan.write_text(json.dumps(value), encoding='utf8')
                with self.assertRaises(ValueError):
                    load_operator_limits(self.seed, {})

    def test_no_plan_keeps_legacy_relative_defaults(self):
        limits = load_operator_limits(self.seed, {'OPERATOR_TRIAL_SECONDS': '3600'})
        self.assertEqual(limits, SessionLimits(seconds=3600))
        for value in (True, 1800.5, '1800.5', 'NaN', '59', '604801'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_operator_limits(self.seed, {'OPERATOR_TRIAL_SECONDS': value})


if __name__ == '__main__':
    unittest.main()
