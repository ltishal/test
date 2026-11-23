import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from quota import QuotaStore, QuotaError


class QuotaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = QuotaStore(Path(self.temp.name) / 'quota.sqlite3',
                                free_turns=2, global_daily_turns=20)
        self.account = 'a' * 32

    def test_first_turn_is_counted_and_ticket_is_once_only(self):
        ticket = self.store.reserve(self.account, 'session1')
        self.assertEqual(self.store.status(self.account)['remaining'], 1)
        self.store.finish(ticket, True, 'Герой получил +50 XP')
        self.assertTrue(self.store.claim_state(self.account, 'session1', ticket, 'Герой получил +50 XP'))
        self.assertFalse(self.store.claim_state(self.account, 'session1', ticket, 'Герой получил +50 XP'))
        self.assertEqual(self.store.report()[0]['completed'], 1)
        self.assertEqual(self.store.report()[0]['started'], 1)

    def test_ticket_is_bound_to_account_session_and_content(self):
        ticket = self.store.reserve(self.account, 'session1')
        self.store.finish(ticket, True, 'секретное сообщение')
        self.assertFalse(self.store.claim_state('b' * 32, 'session1', ticket, 'секретное сообщение'))
        self.assertFalse(self.store.claim_state(self.account, 'session2', ticket, 'секретное сообщение'))
        self.assertFalse(self.store.claim_state(self.account, 'session1', ticket, 'другая реплика'))
        self.assertTrue(self.store.claim_state(self.account, 'session1', ticket, 'секретное сообщение'))

    def test_refund_failed_turn(self):
        ticket = self.store.reserve(self.account, 'session1')
        self.store.finish(ticket, False)
        self.store.finish(ticket, False)
        self.assertEqual(self.store.status(self.account)['remaining'], 2)
        self.assertFalse(self.store.claim_state(self.account, 'session1', ticket, ''))
        self.assertEqual(self.store.report()[0]['started'], 1)
        self.assertEqual(self.store.report()[0]['completed'], 0)

    def test_free_limit_and_pack(self):
        for _ in range(2):
            ticket = self.store.reserve(self.account, 's')
            self.store.finish(ticket, True, 'ok')
        with self.assertRaises(QuotaError):
            self.store.reserve(self.account, 's')
        self.store.grant_pack(self.account, 3)
        self.assertEqual(self.store.status(self.account)['tier'], 'pack')
        ticket = self.store.reserve(self.account, 's')
        self.store.finish(ticket, True, 'ok')
        self.assertEqual(self.store.status(self.account)['pack_left'], 2)

    def test_concurrent_reservations_are_atomic(self):
        def f(_):
            try:
                return self.store.reserve(self.account, 'same')
            except QuotaError as exc:
                return exc.status
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(f, range(3)))
        self.assertEqual(sum(isinstance(x, str) for x in results), 1)
        self.assertEqual(self.store.report()[0]['started'], 1)

    def test_global_cap_and_measured_cost(self):
        store = QuotaStore(Path(self.temp.name) / 'global.sqlite3',
                           free_turns=5, global_daily_turns=1)
        ticket = store.reserve(self.account, 's')
        store.finish(ticket, False)
        with self.assertRaises(QuotaError):
            store.reserve('b' * 32, 's')
        store.record_call(self.account, 'chat', 'fake', 1000000, 200000,
                          input_price=0.15, output_price=0.60)
        report = store.report()[0]
        self.assertAlmostEqual(report['priced_cost_usd'], 0.27, 6)
        self.assertEqual(report['tokens_in'], 1000000)
        self.assertEqual(report['calls'], 1)


if __name__ == '__main__':
    unittest.main()
