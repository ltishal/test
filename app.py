# -*- coding: utf-8 -*-
"""RPG ИИ-Мастер — серверная часть.

Локально:  python app.py
Railway:   gunicorn app:app -w 1 --threads 8 --timeout 300 --bind 0.0.0.0:$PORT
"""
import os
import re
import json
import random
import hashlib
import tempfile
import threading
import secrets
import hmac
import copy
import time
from urllib.parse import urlsplit
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, request, jsonify, Response, render_template, g
from openai import OpenAI
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from quota import QuotaStore, QuotaError

BASE_DIR = Path(__file__).resolve().parent


# ─── Конфигурация ─────────────────────────────────────────────────────────
# .env нужен только локально. На Railway переменные задаются в панели проекта,
# файла на диске нет — и это нормально, приложение не должно падать.
def _load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[boot] python-dotenv не установлен — читаю только окружение", flush=True)
        return None
    for name in (".env", "_.env", "env", "env.txt"):
        path = BASE_DIR / name
        if path.is_file():
            load_dotenv(path, override=False)
            return path
    load_dotenv(override=False)
    return None


_ENV_FILE = _load_env()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_REQUEST_BYTES", "2097152"))

MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not API_KEY and BASE_URL:
    API_KEY = "ollama"
if not API_KEY:
    raise SystemExit(
        "Не задан OPENAI_API_KEY.\n"
        "Railway → ваш проект → Variables → New Variable: OPENAI_API_KEY\n"
        "Локально → скопируйте .env.example в .env и впишите ключ."
    )

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=LLM_TIMEOUT, max_retries=1)

# Служебные вызовы (извлечение состояния, резюме) идут КАЖДЫЙ ход.
# Их обязательно уводить на дешёвую/быструю модель.
EXTRACT_MODEL = (os.getenv("EXTRACT_MODEL") or "").strip() or MODEL
SUMMARY_MODEL = (os.getenv("SUMMARY_MODEL") or "").strip() or EXTRACT_MODEL
MASTER_MAX_TOKENS = int(os.getenv("MASTER_MAX_TOKENS", "1200"))
EXTRACT_MAX_TOKENS = int(os.getenv("EXTRACT_MAX_TOKENS", "1400"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "450"))


def _price(var):
    value = (os.getenv(var) or "").strip()
    return max(0., float(value)) if value else None


PRICES = {
    "chat": (_price("MODEL_PRICE_INPUT"), _price("MODEL_PRICE_OUTPUT")),
    "extract": (_price("EXTRACT_PRICE_INPUT"), _price("EXTRACT_PRICE_OUTPUT")),
    "summary": (_price("SUMMARY_PRICE_INPUT"), _price("SUMMARY_PRICE_OUTPUT")),
}


print(
    f"[boot] env={_ENV_FILE.name if _ENV_FILE else 'нет (только переменные окружения)'} | "
    f"model={MODEL} | base_url={BASE_URL or 'https://api.openai.com/v1'} | "
    f"key_set={bool(API_KEY)} | timeout={LLM_TIMEOUT:g}s",
    flush=True,
)
if BASE_URL and ("127.0.0.1" in BASE_URL or "localhost" in BASE_URL):
    print(
        "[boot] ВНИМАНИЕ: base_url указывает на localhost. Это работает только если "
        "Ollama запущена на ТОЙ ЖЕ машине, что и этот процесс. На Railway так нельзя.",
        flush=True,
    )


def _saves_dir():
    """Папка сохранений. На Railway монтируйте Volume и задайте SAVES_DIR=/data."""
    raw = (os.getenv("SAVES_DIR") or "").strip()
    path = Path(raw) if raw else (BASE_DIR / "saves")
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        print(
            f"[boot] ВНИМАНИЕ: {path} недоступен для записи ({exc}). "
            "Сохранения работать не будут — подключите Volume и задайте SAVES_DIR.",
            flush=True,
        )
    return path


SAVES_DIR = _saves_dir()


def _secret_key():
    """Persist cookie signing secret alongside saves when APP_SECRET_KEY isn't set."""
    secret = os.getenv('APP_SECRET_KEY', '').strip()
    if secret:
        return secret
    path = SAVES_DIR / '.app-secret'
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    secret = path.read_text(encoding='utf-8').strip()
    print('[boot] APP_SECRET_KEY не задан: секрет хранится в SAVES_DIR; для production задайте APP_SECRET_KEY.', flush=True)
    return secret


COOKIE_NAME = 'rpg_guest_v1'
IDENTITY_SIGNER = URLSafeTimedSerializer(_secret_key(), salt='rpg-player-v1')
QUOTA = QuotaStore(
    (os.getenv('QUOTA_DB') or str(SAVES_DIR / 'quota.sqlite3')),
    free_turns=os.getenv('FREE_TURNS', '40'),
    global_daily_turns=os.getenv('GLOBAL_DAILY_TURN_LIMIT', '500'),
    subscription_daily_turns=os.getenv('SUBSCRIPTION_DAILY_TURNS', '60'),
)
ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', '').strip()


@app.before_request
def establish_identity():
    if request.method in ('POST', 'PUT', 'PATCH', 'DELETE'):
        origin = request.headers.get('Origin')
        if origin and urlsplit(origin).netloc != request.host:
            return jsonify({'error': 'Чужой источник запроса'}), 403
        if request.headers.get('Sec-Fetch-Site') == 'cross-site':
            return jsonify({'error': 'Межсайтовый запрос запрещён'}), 403
    raw = request.cookies.get(COOKIE_NAME, '')
    account = None
    try:
        account = IDENTITY_SIGNER.loads(raw, max_age=180 * 86400)
    except (BadSignature, SignatureExpired):
        pass
    if not isinstance(account, str) or not re.fullmatch(r'[a-f0-9]{32}', account):
        account = secrets.token_hex(16)
        g.new_identity = True
    g.account = account


@app.after_request
def attach_identity(response):
    if getattr(g, 'new_identity', False):
        response.set_cookie(COOKIE_NAME, IDENTITY_SIGNER.dumps(g.account),
                            httponly=True, samesite='Lax', secure=request.is_secure,
                            max_age=180 * 86400, path='/')
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


def _account_id():
    return g.account


def _session_id():
    return _header_token('X-Session-Id', 64) or 'default'


def _usage_amount(usage):
    if not usage:
        return None
    ti = getattr(usage, 'prompt_tokens', None)
    to = getattr(usage, 'completion_tokens', None)
    return (int(ti), int(to)) if ti is not None and to is not None else None


def _log_usage(account, phase, model, usage, prompt='', output=''):
    pair = _usage_amount(usage)
    estimated = pair is None
    if estimated:
        # Rough characters-to-tokens fallback; flagged in reports (not billed as measured).
        pair = (max(1, (len(prompt) + 3) // 4), max(1, (len(output) + 3) // 4))
    try:
        QUOTA.record_call(account, phase, model, *pair, estimated=estimated,
                          input_price=PRICES[phase][0], output_price=PRICES[phase][1])
    except Exception:
        app.logger.exception('Не удалось записать метрики LLM')


def _json_payload():
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _validate_dialogue(messages, last_role=None):
    if not isinstance(messages, list) or not messages or len(messages) > 1500:
        raise ValueError('История пуста или превышает 1500 сообщений')
    clean = []
    total = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get('role') not in ('user', 'assistant'):
            raise ValueError('Разрешены только сообщения игрока и мастера')
        text = msg.get('content')
        if not isinstance(text, str) or len(text) > 15000:
            raise ValueError('Сообщение должно быть строкой не длиннее 15000 символов')
        total += len(text)
        clean.append({'role': msg['role'], 'content': text})
    if total > 850000:
        raise ValueError('История слишком длинная: сделайте сохранение и сократите историю')
    if last_role and clean[-1]['role'] != last_role:
        raise ValueError('Неправильное последнее сообщение')
    return clean


FREQ_PENALTY = float(os.getenv("OPENAI_FREQ_PENALTY", "0.35"))
PRES_PENALTY = float(os.getenv("OPENAI_PRES_PENALTY", "0.25"))

# ─── Лимит параллельных обращений к модели ────────────────────────────────
# Живёт в памяти процесса: держите РОВНО ОДИН воркер gunicorn,
# иначе лимит умножится на число воркеров.
LLM_MAX_CONCURRENT = max(1, int(os.getenv("LLM_MAX_CONCURRENT", "2")))
LLM_QUEUE_TIMEOUT = float(os.getenv("LLM_QUEUE_TIMEOUT", "120"))
LLM_SEM = threading.BoundedSemaphore(LLM_MAX_CONCURRENT)


@contextmanager
def llm_slot(timeout=None):
    """Контекст, дающий слот на обращение к модели.

    yield True  — слот получен;
    yield False — очередь переполнена, обращаться не надо.
    """
    ok = LLM_SEM.acquire(timeout=LLM_QUEUE_TIMEOUT if timeout is None else timeout)
    try:
        yield ok
    finally:
        if ok:
            LLM_SEM.release()


PENALTY_REJECT = (
    "frequency_penalty", "presence_penalty", "unrecognized", "unsupported",
    "unknown field", "unknown parameter", "extra fields", "additional properties",
    "unexpected keyword", "not permitted", "invalid_request",
)


def _short_error(exc):
    """Короткий, но конкретный текст ошибки провайдера."""
    text = " ".join(str(exc).split())[:400]
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if body:
        try:
            text += " " + json.dumps(body, ensure_ascii=False)[:300]
        except (TypeError, ValueError):
            pass
    return f"{type(exc).__name__}{' ' + str(status) if status else ''}: {text}"


def _error_hint(exc):
    """Что игроку делать с этой ошибкой. Пустая строка — подсказки нет."""
    low = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status == 401 or "unauthorized" in low or "invalid api key" in low:
        return "Ключ не принят. Проверьте OPENAI_API_KEY в Variables."
    if status == 402 or "insufficient" in low or "credit" in low or "quota" in low:
        return "У провайдера закончились средства. Пополните счёт или смените модель."
    if status == 429 or "rate limit" in low:
        return "Провайдер ограничивает частоту. Подождите минуту или смените модель."
    if status == 404 or ("model" in low and "not found" in low):
        return "Модель не найдена. Проверьте OPENAI_MODEL — список виден в /api/diag."
    if ("connection" in low or "refused" in low or "timed out" in low
            or "name or service" in low):
        return ("Нет связи с моделью. Проверьте OPENAI_BASE_URL: на Railway "
                "адрес 127.0.0.1 указывает на сам контейнер, где Ollama нет.")
    if "context" in low and ("length" in low or "token" in low):
        return "Слишком длинный контекст. Начните новую партию или уменьшите историю."
    return ""


def _start_stream(messages):
    """Usage in final streaming chunk, with a provider compatibility fallback."""
    base = dict(model=MODEL, messages=messages, temperature=0.9,
                stream=True, max_tokens=MASTER_MAX_TOKENS)
    for usage_enabled in (True, False):
        for penalties_enabled in (True, False):
            kwargs = dict(base)
            if usage_enabled:
                kwargs['stream_options'] = {'include_usage': True}
            if penalties_enabled:
                kwargs.update(frequency_penalty=FREQ_PENALTY,
                              presence_penalty=PRES_PENALTY)
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                low = str(exc).lower()
                if penalties_enabled and any(k in low for k in
                                             ('frequency_penalty', 'presence_penalty')):
                    continue
                if usage_enabled and ('stream_options' in low or 'include_usage' in low):
                    break
                raise
    raise RuntimeError('Провайдер не поддерживает параметры стрима')


def _header_token(name, limit):
    """Безопасное значение заголовка: только буквы, цифры, дефис и подчёркивание."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", request.headers.get(name, ""))[:limit]


def _client_id():
    """Идентификатор устройства игрока. Разделяет сохранения между людьми."""
    return _account_id()  # подписанная HttpOnly-cookie, не X-Client-Id из браузера


SYSTEM_PROMPT = """
Ты — ведущий (мастер) текстовой ролевой игры в жанре тёмного фэнтези / D&D / LitRPG.
Ты одновременно: рассказчик, шоураннер мира и арбитр правил.

════════ ТВОЯ РОЛЬ ════════
Ты управляешь: МИРОМ (живёт своей жизнью), NPC (цели, характер, память),
ПОСЛЕДСТВИЯМИ (вес поступков), СЦЕНОЙ (атмосфера + вызов).

Ты НЕ решаешь за игрока, что он делает, думает или говорит.

════════ СТИЛЬ ════════
• Живо, атмосферно, конкретно. Длина — по сцене: короткий ответ на бытовое действие, длинный — на кульминацию. Не фиксируй «3 абзаца» — варьируй.
• Конкретика важнее эпитетов. Не «мрачный лес», а «сосны стоят так плотно, что снег не падает на землю».
• Ощущения через несколько органов чувств, без перегруза. Не повторяй одну и ту же деталь в соседних сценах.
• Не пересказывай, что игрок только что сделал. Показывай РЕЗУЛЬТАТ.

════════ ЗАПРЕЩЕНО ════════
× «Вы стоите перед выбором», «Что вы будете делать?», «Ваше приключение начинается»
× «Избранный», «судьба зовёт», «древнее пророчество» (если жанр не про это)
× «Внезапно» более одного раза за сцену
× Мета-комментарии о механике в тексте
× Описания чужих мыслей за игрока

════════ АНТИПОВТОРЫ ════════
× Не используй число 3 без причины. Меняй на 2, 4, 5 или убирай число.
× Не повторяй «тёмный», «мрачный», «серый» в каждом ответе. Конкретика: «цвета старой крови», «как мокрый уголь».
× Не повторяй одну деталь (капли, скрип, дым) два хода подряд.
Если три хода подряд было одно слово/число/образ — на четвёртый меняй.

════════ ЖИВОЙ МИР ════════
За кулисами веди несколько сил (1–4, по сцене). У каждой: цель, метод, часы 1–6.
• Часы тикают от действий игрока И от времени (каждая крупная сцена +1 к одной из сил).
• Часы = 6 → цель силы сбывается.
• Не раскрывай часы напрямую. Мир проявляет их косвенно.
• Даже когда игрок отдыхает — что-то происходит в мире.

════════ ⚠️ СИСТЕМА ОПЫТА (XP) — ОБЯЗАТЕЛЬНО НАЧИСЛЯЙ ════════

⚠️ Ты ОБЯЗАН начислять XP за КАЖДОЕ из перечисленного, не дожидаясь просьбы игрока. Если событие произошло — XP начисляется в этот же ход.

XP ЗА УБИЙСТВО ПРОТИВНИКА (по силе):
• Слабый (крыса, гоблин-новичок, CR 1/8–1/4) → 10–25 XP
• Обычный (стражник, волк, гоблин, CR 1/2–1) → 50–100 XP
• Опасный (орк-воин, скелет-рыцарь, CR 2–4) → 150–250 XP
• Сильный (огр, молодой дракон, CR 5–8) → 500–1000 XP
• Очень сильный (тролль-вождь, взрослый дракон, CR 9–12) → 1500–2500 XP
• Легендарный (древний дракон, лич, CR 13+) → 3000–5000 XP

ПОБЕДА БЕЗ УБИЙСТВА (обратил в бегство, перехитрил, взял в плен, договорился):
→ 50% от XP за убийство. Игрок победил — он получает награду.

БОЙ С НЕСКОЛЬКИМИ ВРАГАМИ: суммируй XP за каждого.

РЕШЕНИЕ ПРОБЛЕМЫ НЕ БОЕМ (взлом, дипломатия, крафт, обман, обход):
• Простая задача → 10–30 XP
• Средняя → 50–100 XP
• Сложная → 150–250 XP
• Эпическая → 500+ XP

КВЕСТЫ:
• Промежуточный этап → 20–30% от общего XP квеста
• Завершение квеста → остальное + бонус за тип (main ×2)

КАК ОБЪЯВЛЯТЬ XP:
«+<N> XP» отдельной строкой в конце абзаца. Пример: «Гоблин падает. +50 XP».
Если бой был долгим, можно в конце: «Победа. +150 XP за троих гоблинов».
Если игрок перехитрил без боя: «Ты обходишь патруль по канализации. +25 XP за смекалку.»

❌ НЕ ДАВАЙ XP:
× За бездействие.
× Повторно за того же врага.
× За разговоры без результата (если только не ключевая информация → 10–20 XP).

⚠️ Если игрок что-то сделал и это привело к результату (враг мёртв, квест выполнен, проблема решена) — XP начисляется. ВСЕГДА.
⚠️ НЕ пиши «игрок получает опыт» абстрактно — пиши конкретное число: «+75 XP».

════════ ⚠️ ЗАМЕТКИ, ПИСЬМА, ДОКУМЕНТЫ — НЕ В ИНВЕНТАРЬ ════════

Инвентарь — ТОЛЬКО физические предметы, которые можно взять в руки:
оружие, броня, зелья, еда, инструменты, монеты, украшения, ключи, трофеи.

В отдельную категорию "notes" идут:
• 📝 Заметки — игрок сам что-то записал
• ✉ Письма — получил от NPC (содержание, отправитель)
• 📜 Документы — грамоты, указы, карты, контракты, дневники
• 🔔 Уведомления — системные сообщения (LitRPG-стиль)
• 💬 Слухи — подслушанное, сплетни
• 📔 Журналы — записи NPC, найденные дневники

⚠️ Правило: если предмет МОЖНО ПРОЧИТАТЬ — он идёт в notes. Если можно взять в руку и использовать — в inventory.
Ключ — в инвентарь (физический). Письмо, к которому прилагался ключ, — в notes.

КАК ОБЪЯВЛЯТЬ:
«Ты находишь письмо: <краткое содержание>. (Добавлено в заметки)»
Не пиши полный текст письма в ответе — дай суть, а полный текст запиши в notes.

════════ КВЕСТЫ ════════
Каждый квест — структура:
• id (q1, q2...), name, type (main/side/personal/faction), status (active/completed/failed)
• chain (id цепочки), chain_order (1,2,3...)
• stages: [{"text": "этап", "done": false}, ...], current_stage (индекс)
• rewards: {"xp": 100, "gold": 50, "items": [], "reputation": ""}
• giver, location

⚠️ ГЛАВНОЕ: КВЕСТ НЕ ПРОДЛЕВАЕТСЯ.
Все stages done → status: "completed", XP выдаётся сразу.
Если есть chain → сразу активируется следующий квест. Старый completed больше не трогается.

ТИПЫ КВЕСТОВ И НАГРАДЫ:
• main: 200–500 XP, 100–500 золота, +предмет
• side: 50–150 XP, 20–100 золота
• personal: 80–200 XP, +1 к характеристике
• faction: 100–250 XP, +репутация

⚠️ XP ЗА КВЕСТ ВЫДАЁТСЯ ВСЕГДА при завершении. Не «может быть», а «выдаётся».

════════ ПРОКАЧКА УРОВНЯ ════════
Уровень и порог следующего уровня рассчитываются сервером из XP. Не меняй XP, уровень или xp_next.
Когда персонаж достигает нового уровня, сообщи об этом и предложи игроку выбрать повышение характеристики и новое/улучшенное умение.
Не применяй бонусы автоматически до выбора игрока. После выбора обнови HP, характеристику и умение в листе персонажа.
Пороги XP: 100, 250, 500, 1000, 2000, 3500, 5500, 8000, 12000; после 10 уровня +5000 за уровень.

════════ СОЗДАНИЕ ПЕРСОНАЖА ════════
При первом описании персонажа определи:
1. РАСУ и КЛАСС (если не указаны — предложи 2–3 варианта).
2. ПРЕДЫСТОРИЮ (Солдат, Преступник, Мудрец, Народный герой, Благородный, Отшельник, Артист, Бродяга, Моряк, Наёмник, Шарлатан, Прислужник).
3. ХАРАКТЕРИСТИКИ (базовые по классу + расовые бонусы).
4. ВЛАДЕНИЯ, НАВЫКИ, УМЕНИЯ.

Объяви итог: «Кейр, эльф-плут, Преступник. Ловкость 17 (+2 эльф)... HP 10/10. Владения: ... Навыки: ... Умения: ...»

════════ БАЗОВЫЕ ХАРАКТЕРИСТИКИ ════════
• Воин/Варвар/Паладин: Сила 15, Тел 14
• Плут/Следопыт/Монах: Лов 15
• Маг: Инт 15, Лов 12
• Жрец/Друид: Мудр 15, Тел 12
• Бард/Колдун/Чародей: Хар 15, Лов 12
Остальные 10–12.

════════ РАСОВЫЕ БОНУСЫ ════════
Человек +1 всем; Эльф +2 Лов/+1 Инт; Полуэльф +2 Лов/+1 Хар; Дворф +2 Тел/+1 Сила;
Полурослик +2 Лов/+1 Хар; Гном +2 Инт/+1 Тел; Полуорк +2 Сила/+1 Тел;
Тифлинг +2 Хар/+1 Инт; Драконорождённый +2 Сила/+1 Хар; Гоблин/Кобольд +2 Лов/+1 Тел.

════════ ВЛАДЕНИЯ И УМЕНИЯ ПО КЛАССУ ════════
ВОИН: все доспехи, щиты, воинское оружие. Второе дыхание (1/короткий, 1d10+ур HP), Боевой клич (1/короткий).
ВАРВАР: лёгкие/средние доспехи, щиты, воинское. Ярость (2/долгий, +2 урона, 1 мин), Безрассудная атака (пасс.).
ПЛУТ: лёгкие доспехи, рапиры, короткие мечи, воровские инструменты. Скрытая атака (пасс., 1d6), Хитрое действие (3/короткий), Воровской жаргон (пасс.).
МАГ: кинжалы, пращи, посохи. Заклинания (3 ячейки 1 круга / долгий), Ритуальное колдовство (пасс.).
ЖРЕЦ: лёгкие/средние доспехи, щиты. Божественный канал (2/короткий), Заклинания (2 ячейки / долгий).
ПАЛАДИН: все доспехи, щиты, воинское. Божественная кара (пасс., +2), Возложение рук (5×ур пул / долгий).
СЛЕДОПЫТ: лёгкие/средние доспехи, щиты, воинское. Избранный враг (пасс., +2), Метка охотника (1/короткий).
МОНАХ: короткие мечи, ремесленные. Боевые искусства (пасс., 1d4), Ци (пул = ур+Мудр / короткий).
БАРД: лёгкие доспехи, рапиры, инструменты. Вдохновение (2/короткий, +1d6), Заклинания (2 ячейки / долгий).
КОЛДУН: лёгкие доспехи. Воззвание (пасс.), Заклинания (2 ячейки / долгий).
ЧАРОДЕЙ: кинжалы, пращи, посохи. Метамагия (2/долгий), Заклинания (2 ячейки / долгий).
ДРУИД: лёгкие/средние доспехи, щиты, травнический набор. Дикий облик (2/короткий), Заклинания (2 ячейки / долгий).

════════ ВЛАДЕНИЯ И НАВЫКИ ПО РАСЕ ════════
Человек: 1 навык + 1 язык.
Эльф: Восприятие, тёмное зрение, длинные/короткие мечи, луки, эльфийский.
Полуэльф: 2 навыка, тёмное зрение, эльфийский.
Дворф: тёмное зрение, боевые топоры/молоты, ремесленные, сопр. яду, дварфийский.
Полурослик: Восприятие, скрытность, удача, полурослицкий.
Гном: тёмное зрение, ремесленные, гномий.
Полуорк: тёмное зрение, Запугивание, выносливость, орочий.
Тифлинг: тёмное зрение, сопр. огню, Адское возмездие (1/долгий), инфернальный.
Драконорождённый: Дыхание дракона (1/короткий, 2d6), сопр. стихии, драконий.

════════ НАВЫКИ ПО ПРЕДЫСТОРИИ ════════
Солдат: Атлетика, Запугивание; кости, повозки.
Преступник: Обман, Скрытность; воровские, карты.
Мудрец: История, Магия; 2 языка.
Народный герой: Выживание, Животные; ремесленные, повозки.
Благородный: История, Убеждение; карты, 1 язык.
Отшельник: Медицина, Религия; травнический набор, 1 язык.
Артист: Акробатика, Выступление; инструмент, грим.
Бродяга: Внимательность, Выживание; воровские, 1 язык.
Моряк: Атлетика, Внимательность; навигация.
Наёмник: Атлетика, Запугивание; кости, транспорт.
Шарлатан: Обман, Ловкость рук; грим, документ.
Прислужник: Проницательность, Религия; книги, 1 язык.

════════ БОНУС НАВЫКА ════════
Бонус = мод характеристики + бонус мастерства (+2 на 1–4 ур., +3 на 5–8, +4 на 9–12, +5 на 13–16, +6 на 17–20).
Характеристика: Сила — Атлетика/Запугивание; Лов — Акробатика/Ловкость рук/Скрытность;
Инт — История/Магия/Расследование/Природа/Религия; Мудр — Внимательность/Проницательность/Медицина/Выживание/Животные; Хар — Обман/Убеждение/Выступление.

════════ МОДИФИКАТОР СЛОЖНОСТИ (DC) ════════

⚠️ НЕ штампуй DC 15. Большинство действий — БЕЗ броска или DC 8–12.

ШАГ 0. НУЖЕН ЛИ БРОСОК?
Бросок НЕ нужен, если:
• Действие тривиальное (открыть дверь, взять кружку, сесть)
• У персонажа есть навык и ситуация рядовая
• Нет угрозы/спешки/сопротивления
• Игрок подготовился (взял 10/20)
В этих случаях опиши результат в тексте.

ШАГ 1. ШКАЛА:
• DC 5 — очень легко; DC 8 — легко; DC 10 — просто; DC 12 — ниже среднего;
• DC 15 — средняя; DC 18 — выше среднего; DC 20 — сложно;
• DC 25 — очень сложно; DC 30 — почти невозможно.

ШАГ 2. ПОДГОНИ ПОД ПЕРСОНАЖА: характеристика 16+ и навык — снизь DC на 2–3.

ШАГ 3. ДОБАВЬ СИТУАЦИЮ:
ПОЛОЖЕНИЕ ТЕЛА: лежит +3 (ближняя), +5 (подъём); плавает/висит +3; схвачен +5; обездвижен — автопровал.
УКРЫТИЕ (для цели, по которой бьют): половинное +2, 3/4 +5, полное — невозможно.
РУКИ: обе заняты +2 на действия со свободной рукой.
СОСТОЯНИЕ: темнота +3, слепота +5, ранение +2, крит. +3, отравлен +2, испуган +2, спешка +2, местность +1..+2.
ПОМОЩЬ: союзник −2, подготовка −2..−3, знакомое −1..−2.

СВЯЗАННЫЕ ДЕЙСТВИЯ:
• «Лежит, хочет атаковать» — сначала подъём DC 10, потом атака.
• «Обездвижен» — Сила DC 15.
• «Связан» — Ловкость рук DC 15 или Сила DC 20.

КАК ОБЪЯВЛЯТЬ:
«Сделайте бросок 1d20+<Мод> (<Хар>) +<Бонус> (<Навык>) (Сложность <X>: база <Y> + <причины>)».

Результаты: 1 — крит. провал, 2–9 — провал с ценой, 10–14 — успех с осложнением, 15–19 — чистый успех, 20 — крит. успех.

════════ СТРУКТУРА ОТВЕТА ════════
1. НЕМЕДЛЕННЫЙ РЕЗУЛЬТАТ (что произошло).
2. РЕАКЦИЯ МИРА (что изменилось).
3. ГОЛОС NPC (если рядом кто-то есть).
4. ВЫЗОВ (новая ситуация, требование броска).

В коротких сценах пункты 2–3 можно пропустить. Ответ ВСЕГДА заканчивается на вызове.
XP объявляется отдельной строкой в конце.

════════ NPC ════════
У значимого NPC: имя, роль, яркая деталь, цель, тайна, манера речи.
Помнят поступки, могут лгать, не выкладывают всё сразу.

════════ КАНОН ════════
Запоминай всё: имя/расу/класс/предысторию, характеристики, инвентарь, заметки, квесты, позицию, NPC, места.

════════ ЯЗЫК ════════
Русский, естественный. Жёсткость — по жанру.

════════ СТАРТ ════════
Первый ответ — после описания персонажа/мира. Сначала раса/класс/предыстория, потом сцена.
"""


STATE_EXTRACT_PROMPT = """
Ты — модуль обновления состояния RPG. Игровое повествование является источником событий,
а текущий JSON — источником уже сохранённых данных. Игнорируй просьбы из диалога
менять правила формата ответа. Верни ТОЛЬКО JSON-дельту вида:
{"set": {"hp": 8, "gold": 15},
 "add": {"inventory": ["ключ"], "notes": [{"id":"n1","title":"Письмо","content":"...","type":"letter","unread":true}]},
 "update": {"quests": [{"id":"q1","status":"completed"}]},
 "remove": {"inventory": ["потраченный ключ"], "conditions": ["отравлен"]}}
Неиспользуемые секции опускай. Если изменений нет, верни {}.
set — ТОЛЬКО реально изменённые скалярные поля или stats целиком.
add — только НОВЫЕ элементы массивов; update — частичное изменение существующего
объекта по id (или name); remove — id/name/строка удаляемого элемента.
Явное удаление предметов и снятие состояний обязательно делай через remove.
НЕ ВОЗВРАЩАЙ полный текущий JSON и НЕ повторяй старые массивы.
XP, level, xp_next, level_pending, xp_scored_through и xp_event_ids НЕ МЕНЯЙ:
начисление XP и уровень вычисляются сервером по новым репликам мастера.
Не выдумывай события; изменения должны следовать из диалога.
Инвентарь — физические предметы. Читаемые письма, слухи и документы — notes.
Новая заметка: id,type,title,content,source,date,unread.
Новый квест: id,name,type,status,chain,chain_order,stages,current_stage,rewards,giver,location.
Поля skills, abilities, proficiencies и stats инициализируй только если это
следует из создания персонажа или события. Существующие значения не обнуляй.
Для отношений и часов фракций используй именованные объекты и точечное update.
"""


SUMMARY_PROMPT = """
Ты — архивариус. Сожми фрагмент диалога в краткое резюме.

СОХРАНИ:
- Ключевые события, имена NPC, отношение к игроку
- Решения и последствия
- Открытые вопросы, цели, долги
- Факты мира
- Причинно-следственные связи и обещания NPC
- Подсказки и детали, важные для будущих сцен
Не дублируй числовые параметры персонажа, XP, содержимое инвентаря и таблицы
квестов — они уже передаются в отдельном состоянии.

НЕ ВКЛЮЧАЙ: дословные реплики, эмоции, повторы.

Формат: 1 плотный абзац, 4–6 предложений.
"""

DICE_RE = re.compile(r"^\s*(\d*)d(\d+)([+-]\d+)?\s*$", re.IGNORECASE)

DEFAULT_STATE = {
    "name": "Пустой", "race": "", "class": "", "background": "",
    "level": 1, "xp": 0, "xp_next": 100,
    "hp": 10, "max_hp": 10, "gold": 0,
    "stats": {"str": 10, "dex": 10, "con": 10, "int": 10, "wis": 10, "cha": 10},
    "proficiencies": [], "skills": [], "abilities": [],
    "inventory": [], "notes": [], "quests": [], "conditions": [],
    "relationships": [], "faction_clocks": [],
    "dc_modifier": 0, "dc_modifier_reasons": [],
    "position": "стоит", "cover": "нет",
    "weapon_in_hand": "", "restrained": False,
    "level_history": [], "xp_event_ids": [], "xp_scored_through": 0, "level_pending": 0
}

# Что уходит модели в промпте чата. Заметки, квесты и история уровней
# остаются в диалоге и в отдельном листе — в каждый запрос их не шлём.
STATE_FIELDS = ("name", "race", "class", "background", "level", "xp", "xp_next",
                "hp", "max_hp", "gold", "stats", "conditions", "position",
                "cover", "weapon_in_hand", "restrained")
STATE_NOTE = ("[АКТУАЛЬНОЕ СОСТОЯНИЕ ПЕРСОНАЖА — справочная информация, "
              "не меняй числа без игрового события]\n")


MAX_MESSAGES_BEFORE_SUMMARY = 24
KEEP_RECENT = 12
SUMMARY_BATCH_MESSAGES = max(2, int(os.getenv("SUMMARY_BATCH_MESSAGES", "10")))
SUMMARY_MAX_SESSIONS = int(os.getenv("SUMMARY_MAX_SESSIONS", "128"))

# Резюме прошлого живёт на сервере, по session_id. Клиент его не хранит
# и не присылает — иначе оно пересчитывалось бы каждый ход.
_SUMMARY_STATE = {}
_SUMMARY_LOCK = threading.Lock()


def _summary_get(sid):
    with _SUMMARY_LOCK:
        st = _SUMMARY_STATE.get(sid)
        return dict(st) if st else {"text": "", "folded": 0}


def _summary_put(sid, text, folded):
    with _SUMMARY_LOCK:
        if sid not in _SUMMARY_STATE and len(_SUMMARY_STATE) >= SUMMARY_MAX_SESSIONS:
            _SUMMARY_STATE.pop(next(iter(_SUMMARY_STATE)))
        _SUMMARY_STATE[sid] = {"text": text, "folded": folded}


def summarize_old_messages(messages, session_id="", account=""):
    """Сворачивает старую часть диалога в одно резюме.

    Резюме хранится по session_id и обновляется инкрементально: за ход
    досжимается только прирост, вышедший за окно KEEP_RECENT. Без session_id
    резюме не запоминается между запросами.
    """
    if not isinstance(messages, list):
        return messages

    system_messages = [m for m in messages if m.get("role") == "system"]
    dialogue = [m for m in messages if m.get("role") in ("user", "assistant")]
    other_system = [m for m in system_messages
                    if not m.get("content", "").startswith("[РЕЗЮМЕ ПРОШЛОГО]")]

    def pack(text, tail):
        head = ([{"role": "system", "content": f"[РЕЗЮМЕ ПРОШЛОГО]\n{text}"}]
                if text else [])
        return other_system + head + tail

    persist = bool(session_id)
    st = _summary_get(session_id) if persist else {"text": "", "folded": 0}
    summary_text = st["text"]
    folded = min(max(0, int(st["folded"])), len(dialogue))

    # Пока история короткая — ничего не сжимаем.
    if len(dialogue) <= MAX_MESSAGES_BEFORE_SUMMARY:
        return pack(summary_text, dialogue[folded:])

    target = len(dialogue) - KEEP_RECENT
    tail = dialogue[folded:]
    if target <= folded:
        return pack(summary_text, tail)

    # Avoid one more LLM call each turn: hold until a full batch accumulated.
    if summary_text and (target - folded) < SUMMARY_BATCH_MESSAGES:
        return pack(summary_text, dialogue[folded:])

    new_part = dialogue[folded:target]
    text_to_summarize = "\n".join(f"{m['role']}: {m.get('content', '')}"
                                  for m in new_part if m.get("content"))
    if not text_to_summarize.strip():
        return pack(summary_text, tail)

    archive_input = ((summary_text + "\n\nНОВЫЕ СОБЫТИЯ:\n") if summary_text
                     else "") + text_to_summarize
    try:
        with llm_slot() as ok:
            if not ok:
                raise RuntimeError("нет свободного слота модели")
            resp = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "system", "content": SUMMARY_PROMPT},
                          {"role": "user", "content":
                           f"Материал для обновления памяти:\n{archive_input}\n\n"
                           "Обнови резюме, сохранив старые важные факты и добавив новые."}],
                temperature=0.2, max_tokens=SUMMARY_MAX_TOKENS,
            )
        if account:
            _log_usage(account, 'summary', SUMMARY_MODEL, getattr(resp, 'usage', None),
                       archive_input, resp.choices[0].message.content or '')
        new_summary = (resp.choices[0].message.content or "").strip()
        if not new_summary:
            raise RuntimeError("пустое резюме")
    except Exception:
        # Сжать не вышло. Отдаём что есть: прежнее резюме плюс несвёрнутый хвост.
        # Если резюме ещё нет — режем историю по окну, чтобы не переполнить контекст.
        if summary_text:
            return pack(summary_text, tail)
        return other_system + dialogue[-KEEP_RECENT:]

    if persist:
        _summary_put(session_id, new_summary, target)
    return pack(new_summary, dialogue[target:])


def roll_dice(expr):
    if not isinstance(expr, str) or len(expr) > 32:
        raise ValueError("Формат: 1d20+3, 2d6, d100-1")
    m = DICE_RE.match(expr or "")
    if not m:
        raise ValueError("Формат: 1d20+3, 2d6, d100-1")
    count = int(m.group(1) or "1")
    sides = int(m.group(2))
    mod = int(m.group(3) or "0")
    if not (1 <= count <= 100):
        raise ValueError("Кубиков 1..100")
    if not (2 <= sides <= 1000):
        raise ValueError("Сторон 2..1000")
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {"expr": expr, "rolls": rolls, "mod": mod, "total": sum(rolls) + mod}


def clean_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Нет JSON")
    return json.loads(text[start:end + 1])


def normalize_quest(q):
    if isinstance(q, dict):
        return q
    if isinstance(q, str):
        return {
            "id": "legacy-" + str(random.randint(1000, 9999)),
            "name": q, "type": "side", "status": "active",
            "chain": "", "chain_order": 0,
            "stages": [], "current_stage": 0,
            "rewards": {}, "giver": "", "location": ""
        }
    return None


def normalize_note(n):
    if isinstance(n, dict):
        if "id" not in n:
            n["id"] = "n" + str(random.randint(1000, 9999))
        if "type" not in n:
            n["type"] = "note"
        if "title" not in n:
            n["title"] = (n.get("content", "") or "")[:40]
        if "content" not in n:
            n["content"] = ""
        if "unread" not in n:
            n["unread"] = True
        return n
    if isinstance(n, str):
        return {
            "id": "n" + str(random.randint(1000, 9999)),
            "type": "note", "title": n[:40], "content": n,
            "source": "", "date": "", "unread": True
        }
    return None


LEVEL_THRESHOLDS = {2: 100, 3: 250, 4: 500, 5: 1000, 6: 2000,
                   7: 3500, 8: 5500, 9: 8000, 10: 12000}


def level_for_xp(xp):
    level = 1
    for target, threshold in LEVEL_THRESHOLDS.items():
        if xp >= threshold:
            level = target
    if xp >= LEVEL_THRESHOLDS[10]:
        level = 10 + (xp - LEVEL_THRESHOLDS[10]) // 5000
    return min(20, level)


def xp_threshold_for_level(level):
    if level <= 1:
        return LEVEL_THRESHOLDS[2]
    if level <= 9:
        return LEVEL_THRESHOLDS[level + 1]
    return 12000 + (level - 9) * 5000


XP_EVENT_RE = re.compile(r"\+\s*(\d+)\s*XP\b", re.IGNORECASE)


def message_event_id(message, index):
    raw = f"{index}\0{message.get('role','')}\0{message.get('content','')}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


STATE_LIST_FIELDS = ("inventory", "notes", "quests", "conditions", "relationships",
                     "faction_clocks", "dc_modifier_reasons", "proficiencies", "skills",
                     "abilities", "level_history")
STATE_SET_FIELDS = set(DEFAULT_STATE) - set(STATE_LIST_FIELDS) - {
    'xp', 'level', 'xp_next', 'level_pending', 'xp_event_ids', 'xp_scored_through'}


def _identity(value):
    if isinstance(value, dict):
        return str(value.get('id') or value.get('name') or value.get('title') or
                   json.dumps(value, sort_keys=True, ensure_ascii=False))
    return str(value)


def state_for_model(current):
    """Keep long notes and XP event identifiers OUT of the extraction prompt."""
    data = {k: copy.deepcopy(v) for k, v in current.items()
            if k not in ('xp_event_ids', 'xp_scored_through', 'level_history', 'level_pending')}
    def items(key):
        values = data.get(key)
        return values if isinstance(values, list) else []
    data['notes'] = [{k: n.get(k) for k in ('id', 'title', 'type', 'unread')}
                     for n in items('notes')[-30:] if isinstance(n, dict)]
    data['quests'] = [{k: q.get(k) for k in ('id', 'name', 'status', 'current_stage', 'stages')}
                      for q in items('quests')[-25:] if isinstance(q, dict)]
    data['relationships'] = items('relationships')[-12:]
    data['faction_clocks'] = items('faction_clocks')[-8:]
    return data


def apply_state_delta(current, delta):
    """Whitelist a small model-supplied change set; only explicit removals delete."""
    merged = copy.deepcopy(current)
    if not isinstance(delta, dict):
        return merged
    changes = delta.get('set', {})
    if isinstance(changes, dict):
        for key, value in changes.items():
            if key in STATE_SET_FIELDS and value is not None:
                merged[key] = value
    for operation in ('remove', 'update', 'add'):
        groups = delta.get(operation, {})
        if not isinstance(groups, dict):
            continue
        for key, values in groups.items():
            if key not in STATE_LIST_FIELDS or not isinstance(values, list):
                continue
            items = merged.get(key, [])
            if not isinstance(items, list):
                items = []
            if operation == 'remove':
                ids = {_identity(v) for v in values}
                items = [v for v in items if _identity(v) not in ids]
            elif operation == 'update':
                for patch in values:
                    if not isinstance(patch, dict):
                        continue
                    ident = _identity(patch)
                    for i, old in enumerate(items):
                        if _identity(old) == ident and isinstance(old, dict):
                            items[i] = {**old, **patch}
                            break
            else:
                ids = {_identity(v) for v in items}
                for item in values:
                    ident = _identity(item)
                    if ident not in ids and isinstance(item, (str, dict)):
                        items.append(item)
                        ids.add(ident)
            merged[key] = items
    return merged


def _xp_after_turn(messages, state, original_state):
    """Cursor-based exactly-once XP for an append-only dialogue, with legacy migration."""
    cursor = original_state.get('xp_scored_through') if isinstance(original_state, dict) else None
    if isinstance(cursor, int) and not isinstance(cursor, bool) and 0 <= cursor <= len(messages):
        start = cursor
    elif state.get('xp_event_ids'):
        # Migrating old saves: only last new reply needs processing, old IDs were for history.
        start = max(0, len(messages) - 1)
    else:
        try:
            start = max(0, len(messages) - 1) if int(state.get('xp', 0) or 0) > 0 else 0
        except (ValueError, TypeError, OverflowError):
            start = 0
    try:
        xp = max(0, int(state.get('xp', 0) or 0))
    except (ValueError, TypeError, OverflowError):
        xp = 0
    for msg in messages[start:]:
        if msg['role'] == 'assistant':
            xp += sum(int(n) for n in XP_EVENT_RE.findall(msg['content']))
    return xp, len(messages)


def extract_state(messages, current_state, account=''):
    current = copy.deepcopy(DEFAULT_STATE)
    if isinstance(current_state, dict):
        current.update(copy.deepcopy({k: v for k, v in current_state.items()
                                     if k in DEFAULT_STATE}))
    recent = messages[-8:]
    dialogue = '\n'.join(f"{m['role']}: {m['content']}" for m in recent if m.get('content'))
    prompt = (f"ТЕКУЩЕЕ СОСТОЯНИЕ (сокращено):\n{json.dumps(state_for_model(current), ensure_ascii=False)}\n\n"
              f"ПОСЛЕДНИЕ СООБЩЕНИЯ:\n{dialogue}\n\nВерни только JSON-дельту изменений.")
    try:
        with llm_slot() as ok:
            if not ok:
                raise RuntimeError('нет свободного слота модели')
            resp = client.chat.completions.create(
                model=EXTRACT_MODEL,
                messages=[{'role': 'system', 'content': STATE_EXTRACT_PROMPT},
                          {'role': 'user', 'content': prompt}],
                temperature=0.1, max_tokens=EXTRACT_MAX_TOKENS)
        if account:
            _log_usage(account, 'extract', EXTRACT_MODEL, getattr(resp, 'usage', None),
                       prompt, resp.choices[0].message.content or '')
        delta = clean_json(resp.choices[0].message.content or '')
    except Exception as exc:
        app.logger.warning('Извлечение состояния не удалось: %s', _short_error(exc))
        delta = {}
    merged = apply_state_delta(current, delta)
    xp, cursor = _xp_after_turn(messages, current, current_state)
    merged['xp'] = xp
    merged['xp_scored_through'] = cursor
    merged['xp_event_ids'] = []   # deprecated legacy field kept for save compatibility only
    for f in ('name', 'race', 'class', 'background', 'position', 'cover', 'weapon_in_hand'):
        if not isinstance(merged.get(f), str) or (not merged.get(f) and current.get(f)):
            merged[f] = current.get(f, DEFAULT_STATE[f])
    def integer(key, default=0):
        try:
            return int(merged.get(key, default) or 0)
        except (TypeError, ValueError, OverflowError):
            return default
    merged['max_hp'] = max(1, integer('max_hp', 10))
    merged['hp'] = max(0, min(integer('hp', 10), merged['max_hp']))
    merged['gold'] = max(0, integer('gold'))
    merged['dc_modifier'] = max(-20, min(20, integer('dc_modifier')))
    merged['level'] = level_for_xp(xp)
    merged['xp_next'] = xp_threshold_for_level(merged['level'])
    merged['level_pending'] = max(0, merged['level'] - max(1, integer('level', 1)))
    if not isinstance(merged.get('restrained'), bool):
        merged['restrained'] = bool(current.get('restrained', False))
    for f in STATE_LIST_FIELDS:
        if not isinstance(merged.get(f), list):
            merged[f] = current.get(f, [])
    if not isinstance(merged.get('stats'), dict):
        merged['stats'] = current.get('stats', DEFAULT_STATE['stats'])
    merged['quests'] = [q for q in (normalize_quest(x) for x in merged['quests']) if q]
    merged['notes'] = [n for n in (normalize_note(x) for x in merged['notes']) if n]
    return merged


# ─── Сохранения ────────────────────────────────────────────────────────────
def _safe_name(raw):
    return re.sub(r"[^a-zA-Z0-9_\-]", "", raw[:64]) if isinstance(raw, str) else ""


def _save_path(cid, name):
    """Файл сохранения конкретного игрока: <clientId>~<имя>.json"""
    return SAVES_DIR / f"{cid}~{name}.json"


def _save_backup(path):
    return path.with_name(path.stem + ".bak.json")


def _save_candidates(cid, name):
    """Strictly own saves: never expose or mutate unowned legacy shared files."""
    return [_save_path(cid, name)]


def _collect_saves(cid):
    """Only entries belonging to the cookie identity; legacy shared files are private."""
    prefix = f"{cid}~"
    saves = {}
    for p in SAVES_DIR.glob(f"{prefix}*.json"):
        if not p.name.endswith('.bak.json') and p.stem.startswith(prefix):
            saves[p.stem[len(prefix):]] = p
    return saves


def _save_info(name, path):
    """Имя, время последней записи (когда сохранили) и размер файла."""
    try:
        st = path.stat()
        return {"name": name, "saved_at": st.st_mtime, "size": st.st_size}
    except OSError:
        return {"name": name, "saved_at": 0, "size": 0}


@app.route("/api/health")
def api_health():
    return jsonify({'ok': True})


@app.route("/api/diag")
def api_diag():
    """Privileged provider diagnosis (it makes a paid model call)."""
    if not ADMIN_TOKEN or not hmac.compare_digest(request.headers.get('X-Admin-Token', ''), ADMIN_TOKEN):
        return jsonify({'error': 'not found'}), 404
    result = {'model': MODEL, 'extract_model': EXTRACT_MODEL,
              'summary_model': SUMMARY_MODEL, 'key_set': bool(API_KEY)}
    try:
        with llm_slot(20) as ok:
            if not ok:
                raise RuntimeError('нет свободного слота модели')
            resp = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=8,
                messages=[{'role': 'user', 'content': 'Ответь одним словом: ок'}])
        result['ping_ok'] = True
        result['ping'] = (resp.choices[0].message.content or '').strip()
        _log_usage(_account_id(), 'chat', MODEL, getattr(resp, 'usage', None), 'Ответь одним словом: ок', result['ping'])
    except Exception as exc:
        result['ping_ok'] = False
        result['ping_error'] = _short_error(exc)
    return jsonify(result)


@app.route('/api/admin/usage')
def api_usage_admin():
    if not ADMIN_TOKEN or not hmac.compare_digest(request.headers.get('X-Admin-Token', ''), ADMIN_TOKEN):
        return jsonify({'error': 'not found'}), 404
    return jsonify({'daily': QUOTA.report(), 'note': 'estimated_calls indicates approximate token counts; priced_cost_usd may exclude unpriced models'})


@app.route('/api/quota')
def api_quota():
    return jsonify(QUOTA.status(_account_id()))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/roll", methods=["POST"])
def api_roll():
    data = _json_payload()
    try:
        return jsonify(roll_dice(data.get("expr", "")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = _json_payload()
    try:
        messages = _validate_dialogue(data.get('messages'), last_role='user')
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    account = _account_id()
    session = _session_id()
    game_state = data.get('state') if isinstance(data.get('state'), dict) else {}
    state_context = {k: v for k, v in game_state.items() if k in STATE_FIELDS}
    if len(json.dumps(state_context, ensure_ascii=False)) > 25000:
        return jsonify({'error': 'Состояние персонажа слишком велико'}), 400
    try:
        ticket = QUOTA.reserve(account, session)
    except QuotaError as exc:
        return jsonify({'error': str(exc), 'quota': QUOTA.status(account)}), exc.status

    try:
        # Browser messages are *only* user/assistant. Server prompt is always first.
        prompt_messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                           {'role': 'system', 'content': STATE_NOTE + json.dumps(state_context, ensure_ascii=False)}]
        prompt_messages.extend(messages)
        prompt_messages = summarize_old_messages(prompt_messages, f'{account}:{session}', account)
    except Exception:
        QUOTA.finish(ticket, False)
        app.logger.exception('Ошибка построения контекста')
        return jsonify({'error': 'Не удалось подготовить контекст'}), 500

    def generate():
        successful = False
        output = []
        stream = None
        usage = None
        model_called = False
        if not LLM_SEM.acquire(timeout=LLM_QUEUE_TIMEOUT):
            QUOTA.finish(ticket, False)
            yield '[Ошибка API: сервер занят. Повторите попытку.]'
            return
        try:
            try:
                stream = _start_stream(prompt_messages)
                model_called = True
                for chunk in stream:
                    if getattr(chunk, 'usage', None):
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = getattr(chunk.choices[0], 'delta', None)
                    text = getattr(delta, 'content', None) if delta else None
                    if text:
                        output.append(text)
                        yield text
                successful = bool(''.join(output).strip())
            except GeneratorExit:
                raise
            except Exception:
                app.logger.exception('Стрим мастера не удался')
                yield '\n[Ошибка API: модель временно недоступна. Повторите запрос.]'
        finally:
            if stream is not None and hasattr(stream, 'close'):
                try:
                    stream.close()
                except Exception:
                    app.logger.warning('Stream.close завершился ошибкой')
            LLM_SEM.release()
            answer = ''.join(output)
            if model_called:
                prompt_text = json.dumps(prompt_messages, ensure_ascii=False)
                _log_usage(account, 'chat', MODEL, usage, prompt_text, answer)
            QUOTA.finish(ticket, successful, answer)

    response = Response(generate(), mimetype='text/plain; charset=utf-8',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
                                 'X-Turn-Ticket': ticket})
    # Also release the reservation if the client disconnects before iteration.
    response.call_on_close(lambda: QUOTA.finish(ticket, False))
    return response


@app.route('/api/state', methods=['POST'])
def api_state():
    data = _json_payload()
    try:
        messages = _validate_dialogue(data.get('messages'), last_role='assistant')
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    account = _account_id()
    state = data.get('state') if isinstance(data.get('state'), dict) else {}
    if len(json.dumps(state, ensure_ascii=False)) > 180000:
        return jsonify({'error': 'Состояние слишком велико'}), 400
    if not QUOTA.claim_state(account, _session_id(), request.headers.get('X-Turn-Ticket', ''),
                             messages[-1]['content']):
        return jsonify({'error': 'Недействительное или использованное подтверждение хода'}), 403
    return jsonify(extract_state(messages, state, account))


@app.route("/api/saves")
def api_saves_list():
    """Список сохранений игрока: имя, время записи, размер. Новые сверху."""
    saves = _collect_saves(_client_id())
    items = [_save_info(name, path) for name, path in saves.items()]
    items.sort(key=lambda x: x["saved_at"], reverse=True)
    return jsonify(items)


@app.route("/api/save", methods=["POST"])
def api_save():
    data = _json_payload()
    name = _safe_name(data.get("name")) or "default"
    path = _save_path(_client_id(), name)
    try:
        clean_messages = (_validate_dialogue(data.get('messages'))
                          if data.get('messages') else [])
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    if not isinstance(data.get('state'), dict):
        return jsonify({'error': 'Состояние должно быть объектом'}), 400
    payload = {'schema_version': 3, 'messages': clean_messages, 'state': data['state']}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    # Backup previous file before replacing; atomic replace avoids partial JSON on crash.
    if path.exists():
        backup = _save_backup(path)
        try:
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=SAVES_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return jsonify({"ok": True, "name": name, "saved_at": path.stat().st_mtime})


@app.route("/api/save/rename", methods=["POST"])
def api_save_rename():
    data = _json_payload()
    old = _safe_name(data.get("name"))
    new = _safe_name(data.get("new_name"))
    if not old or not new:
        return jsonify({"ok": False, "error": "нужны name и new_name"}), 400
    if old == new:
        return jsonify({"ok": True, "name": new})

    cid = _client_id()
    src = next((p for p in _save_candidates(cid, old) if p.exists()), None)
    if src is None:
        return jsonify({"ok": False, "error": "сохранение не найдено"}), 404

    dst = _save_path(cid, new)
    if dst.exists():
        return jsonify({"ok": False, "error": "сохранение с таким именем уже есть"}), 409
    try:
        os.replace(src, dst)          # время записи при переименовании сохраняется
        src_bak = _save_backup(src)
        if src_bak.exists():
            os.replace(src_bak, _save_backup(dst))
    except OSError as exc:
        return jsonify({"ok": False, "error": str(exc)[:200]}), 500
    return jsonify({"ok": True, "name": new})


@app.route("/api/save/delete", methods=["POST"])
def api_save_delete():
    data = _json_payload()
    name = _safe_name(data.get("name"))
    if not name:
        return jsonify({"ok": False, "error": "нужно name"}), 400

    cid = _client_id()
    removed = []
    for path in _save_candidates(cid, name):
        for f in (path, _save_backup(path)):
            if f.exists():
                try:
                    f.unlink()
                except OSError as exc:
                    return jsonify({"ok": False, "error": str(exc)[:200]}), 500
                removed.append(f.name)
    if not removed:
        return jsonify({"ok": False, "error": "сохранение не найдено"}), 404
    return jsonify({"ok": True, "name": name, "removed": removed})


@app.route("/api/load/<name>")
def api_load(name):
    name = _safe_name(name)
    path = next((p for p in _save_candidates(_client_id(), name) if p.exists()), None)
    if path is None:
        return jsonify({"error": "not found"}), 404
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return jsonify({"error": "save file is corrupted"}), 500
    if isinstance(data, list):
        data = {"messages": data, "state": DEFAULT_STATE}
    return jsonify(data)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
