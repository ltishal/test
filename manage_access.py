# -*- coding: utf-8 -*-
"""Manual quota administration; no payment or billing integration.

    python manage_access.py report
    python manage_access.py grant <account_id> 300
    python manage_access.py subscribe <account_id> 30
"""
import argparse
import json
import os
import re
from pathlib import Path
from quota import QuotaStore


def main():
    p = argparse.ArgumentParser(description='Администрирование ходов RPG ИИ-Мастер')
    p.add_argument('action', choices=['report', 'grant', 'subscribe'])
    p.add_argument('account', nargs='?')
    p.add_argument('amount', nargs='?', type=int)
    args = p.parse_args()
    root = Path(os.getenv('SAVES_DIR') or Path(__file__).parent / 'saves')
    dbpath = (os.getenv('QUOTA_DB') or str(root / 'quota.sqlite3'))
    quota = QuotaStore(dbpath, os.getenv('FREE_TURNS', '40'),
                       os.getenv('GLOBAL_DAILY_TURN_LIMIT', '500'),
                       os.getenv('SUBSCRIPTION_DAILY_TURNS', '60'))
    if args.action == 'report':
        print(json.dumps(quota.report(), ensure_ascii=False, indent=2))
        return
    if not args.account or not re.fullmatch(r'[0-9a-f]{32}', args.account) or not args.amount:
        p.error('нужны account_id (32 символа hex) и число ходов/дней')
    if args.action == 'grant':
        quota.grant_pack(args.account, args.amount)
        print('Добавлены ходы:', args.amount)
    else:
        quota.grant_subscription(args.account, args.amount)
        print('Подписка активирована, дней:', args.amount)
    print('Статус:', json.dumps(quota.status(args.account), ensure_ascii=False))


if __name__ == '__main__':
    main()
