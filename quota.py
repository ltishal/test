"""SQLite accounting and one-use state tickets for RPG ИИ-Мастер.

All quota changes run in BEGIN IMMEDIATE transactions; never trust browser counters.
Anonymous browser cookies are *not* accounts: clearing cookies resets identity.
Use an invitation gate / real accounts before broad paid deployment.
"""
import hashlib
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path


class QuotaError(Exception):
    def __init__(self, message, status=429):
        super().__init__(message)
        self.status = status


def utc_day():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


class QuotaStore:
    def __init__(self, path, free_turns=40, global_daily_turns=500, subscription_daily_turns=60):
        self.path = str(path)
        self.free_turns = max(0, int(free_turns))
        self.global_daily_turns = max(0, int(global_daily_turns))
        self.subscription_daily_turns = max(1, int(subscription_daily_turns))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA busy_timeout=15000')
        db.execute('PRAGMA foreign_keys=ON')
        return db

    def _init_db(self):
        with self._connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    free_used INTEGER NOT NULL DEFAULT 0,
                    pack_left INTEGER NOT NULL DEFAULT 0,
                    tier TEXT NOT NULL DEFAULT 'free',
                    until_ts REAL NOT NULL DEFAULT 0,
                    pending INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS usage_daily (
                    account TEXT NOT NULL,
                    day TEXT NOT NULL,
                    turns_started INTEGER NOT NULL DEFAULT 0,
                    turns_completed INTEGER NOT NULL DEFAULT 0,
                    tokens_in INTEGER NOT NULL DEFAULT 0,
                    tokens_out INTEGER NOT NULL DEFAULT 0,
                    calls INTEGER NOT NULL DEFAULT 0,
                    estimated_calls INTEGER NOT NULL DEFAULT 0,
                    known_cost_usd REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(account, day)
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket TEXT PRIMARY KEY,
                    account TEXT NOT NULL,
                    session TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    created_ts REAL NOT NULL,
                    started_day TEXT NOT NULL,
                    message_sha TEXT NOT NULL DEFAULT '',
                    charged_as TEXT NOT NULL DEFAULT 'free'
                );
                CREATE INDEX IF NOT EXISTS tickets_by_account ON tickets(account, status);
                CREATE TABLE IF NOT EXISTS model_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account TEXT NOT NULL,
                    day TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    estimated INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL,
                    created_ts REAL NOT NULL
                );
            ''')
            # Do not invalidate live requests when the admin CLI opens the DB.
            self._expire_stale(db)

    def _expire_stale(self, db):
        """Recover streaming requests abandoned by a crashed worker."""
        cutoff = time.time() - 900
        rows = db.execute("SELECT ticket,account,charged_as FROM tickets WHERE status='running' AND created_ts<?",
                          (cutoff,)).fetchall()
        for row in rows:
            db.execute("UPDATE tickets SET status='failed' WHERE ticket=?", (row['ticket'],))
            if row['charged_as'] == 'free':
                db.execute('UPDATE accounts SET free_used=MAX(0,free_used-1) WHERE id=?',
                           (row['account'],))
            elif row['charged_as'] == 'pack':
                db.execute('UPDATE accounts SET pack_left=pack_left+1 WHERE id=?',
                           (row['account'],))
        if rows:
            db.execute("UPDATE accounts SET pending=(SELECT COUNT(*) FROM tickets WHERE tickets.account=accounts.id AND tickets.status='running')")

    def _ensure_account(self, db, account):
        db.execute('INSERT OR IGNORE INTO accounts(id) VALUES(?)', (account,))

    def status(self, account):
        with self._connect() as db:
            self._ensure_account(db, account)
            a = db.execute('SELECT * FROM accounts WHERE id=?', (account,)).fetchone()
            used = db.execute('SELECT turns_started FROM usage_daily WHERE account=? AND day=?',
                              (account, utc_day())).fetchone()
            today = used['turns_started'] if used else 0
            tier = a['tier']
            if tier == 'subscription' and a['until_ts'] > time.time():
                remaining = max(0, self.subscription_daily_turns - today)
            elif a['pack_left'] > 0:
                tier = 'pack'
                remaining = a['pack_left']
            else:
                tier = 'free'
                remaining = max(0, self.free_turns - a['free_used'])
            return {'account_id': account, 'tier': tier, 'remaining': remaining,
                    'free_total': self.free_turns, 'free_used': a['free_used'],
                    'pack_left': a['pack_left'], 'today': today,
                    'subscription_until': a['until_ts'] if tier == 'subscription' else None}

    def reserve(self, account, session):
        ticket = secrets.token_urlsafe(30)
        day = utc_day()
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._expire_stale(db)
            self._ensure_account(db, account)
            a = db.execute('SELECT * FROM accounts WHERE id=?', (account,)).fetchone()
            if a['pending']:
                raise QuotaError('Предыдущий ход ещё обрабатывается.', 409)
            count = db.execute('SELECT SUM(turns_started) FROM usage_daily WHERE day=?',
                               (day,)).fetchone()[0] or 0
            if self.global_daily_turns and count >= self.global_daily_turns:
                raise QuotaError('Суточный лимит сервера исчерпан. Попробуйте завтра.')
            today = db.execute('SELECT turns_started FROM usage_daily WHERE account=? AND day=?',
                               (account, day)).fetchone()
            if a['tier'] == 'subscription' and a['until_ts'] > time.time():
                if (today['turns_started'] if today else 0) >= self.subscription_daily_turns:
                    raise QuotaError('Дневной лимит подписки исчерпан.')
                charged_as = 'subscription'
            elif a['pack_left'] > 0:
                db.execute('UPDATE accounts SET pack_left=pack_left-1 WHERE id=?', (account,))
                charged_as = 'pack'
            elif a['free_used'] < self.free_turns:
                db.execute('UPDATE accounts SET free_used=free_used+1 WHERE id=?', (account,))
                charged_as = 'free'
            else:
                raise QuotaError('Бесплатные ходы закончились. Новые пакеты пока выдаются администратором.')
            db.execute('UPDATE accounts SET pending=pending+1 WHERE id=?', (account,))
            db.execute('''INSERT INTO usage_daily(account,day,turns_started) VALUES(?,?,1)
                       ON CONFLICT(account,day) DO UPDATE SET turns_started=turns_started+1''',
                       (account, day))
            db.execute('''INSERT INTO tickets(ticket,account,session,created_ts,started_day,charged_as)
                          VALUES(?,?,?,?,?,?)''', (ticket, account, session, time.time(), day, charged_as))
        return ticket

    def finish(self, ticket, successful, assistant_text=''):
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            t = db.execute('SELECT * FROM tickets WHERE ticket=?', (ticket,)).fetchone()
            if not t or t['status'] != 'running':
                return
            status = 'ready' if successful and assistant_text.strip() else 'failed'
            digest = hashlib.sha256(assistant_text.encode('utf-8')).hexdigest() if status == 'ready' else ''
            db.execute('UPDATE tickets SET status=?, message_sha=? WHERE ticket=?',
                       (status, digest, ticket))
            db.execute('UPDATE accounts SET pending=MAX(0,pending-1) WHERE id=?', (t['account'],))
            if status == 'ready':
                db.execute('''UPDATE usage_daily SET turns_completed=turns_completed+1
                              WHERE account=? AND day=?''', (t['account'], t['started_day']))
            elif t['charged_as'] == 'free':
                db.execute('UPDATE accounts SET free_used=MAX(0,free_used-1) WHERE id=?', (t['account'],))
            elif t['charged_as'] == 'pack':
                db.execute('UPDATE accounts SET pack_left=pack_left+1 WHERE id=?', (t['account'],))
            # turns_started intentionally remains counted towards the global safety cap.

    def claim_state(self, account, session, ticket, assistant_text):
        if not isinstance(ticket, str) or len(ticket) > 128:
            return False
        digest = hashlib.sha256(assistant_text.encode('utf-8')).hexdigest()
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            changed = db.execute('''UPDATE tickets SET status='state_used' WHERE ticket=? AND account=?
                                    AND session=? AND status='ready' AND message_sha=?''',
                                 (ticket, account, session, digest)).rowcount
        return bool(changed)

    def record_call(self, account, phase, model, tokens_in=None, tokens_out=None,
                    estimated=False, input_price=None, output_price=None):
        # Prices are in USD per million tokens; None = unconfigured, not zero-cost.
        known = (tokens_in is not None and tokens_out is not None and
                 input_price is not None and output_price is not None)
        cost = (tokens_in * input_price + tokens_out * output_price) / 1_000_000 if known else None
        with self._connect() as db:
            db.execute('''INSERT INTO model_calls(account,day,phase,model,tokens_in,tokens_out,
                          estimated,cost_usd,created_ts) VALUES(?,?,?,?,?,?,?,?,?)''',
                       (account, utc_day(), phase, model, tokens_in, tokens_out,
                        int(estimated), cost, time.time()))
            db.execute('''INSERT INTO usage_daily(account,day,tokens_in,tokens_out,calls,estimated_calls,known_cost_usd)
                          VALUES(?,?,?,?,1,?,?) ON CONFLICT(account,day) DO UPDATE SET
                          tokens_in=tokens_in+excluded.tokens_in,
                          tokens_out=tokens_out+excluded.tokens_out,
                          calls=calls+1, estimated_calls=estimated_calls+excluded.estimated_calls,
                          known_cost_usd=known_cost_usd+excluded.known_cost_usd''',
                       (account, utc_day(), tokens_in or 0, tokens_out or 0,
                        int(estimated or tokens_in is None), cost or 0))

    def report(self, limit=30):
        with self._connect() as db:
            rows = db.execute('''SELECT day, SUM(turns_started) started, SUM(turns_completed) completed,
                       SUM(tokens_in) tokens_in, SUM(tokens_out) tokens_out,
                       SUM(calls) calls, SUM(estimated_calls) estimated_calls,
                       SUM(known_cost_usd) priced_cost_usd FROM usage_daily
                       GROUP BY day ORDER BY day DESC LIMIT ?''', (limit,)).fetchall()
            return [dict(x) for x in rows]

    def grant_pack(self, account, turns):
        if not (1 <= turns <= 100000):
            raise ValueError('turns must be 1..100000')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._ensure_account(db, account)
            db.execute('UPDATE accounts SET pack_left=pack_left+? WHERE id=?', (turns, account))

    def grant_subscription(self, account, days):
        if not (1 <= days <= 365):
            raise ValueError('days must be 1..365')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._ensure_account(db, account)
            db.execute("UPDATE accounts SET tier='subscription',until_ts=? WHERE id=?",
                       (time.time() + days * 86400, account))
