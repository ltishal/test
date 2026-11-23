# -*- coding: utf-8 -*-
"""RPG ИИ-Мастер — серверная часть.

Локально: python app.py
Railway: gunicorn -c gunicorn.conf.py app:app
"""
import os
import re
import json
import random
import hashlib
import tempfile
import time
import hmac
import unicodedata
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, request, jsonify, Response, render_template, g
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent


def _load_env():
    """Локальный .env необязателен; переменные платформы имеют приоритет."""
    try:
        from dotenv import load_dotenv
    except ImportError:
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

# Ограничение размера JSON-запроса защищает сервер от случайно огромных payload.
app.config["MAX_CONTENT_LENGTH"] = max(
    262144, int(os.getenv("RPG_MAX_REQUEST_BYTES", str(4 * 1024 * 1024)))
)

LLM_TIMEOUT = max(15.0, float(os.getenv("LLM_TIMEOUT", "300")))
LLM_MAX_CONCURRENT = max(1, min(16, int(os.getenv("LLM_MAX_CONCURRENT", "2"))))
LLM_QUEUE_TIMEOUT = max(1.0, min(900.0, float(os.getenv("LLM_QUEUE_TIMEOUT", "120"))))
LLM_SEM = threading.BoundedSemaphore(LLM_MAX_CONCURRENT)

# Provider configuration. Yandex AI Studio takes priority when its variables are set.
YANDEX_API_KEY = (os.getenv("YANDEX_API_KEY") or "").strip()
YANDEX_FOLDER_ID = (os.getenv("YANDEX_FOLDER_ID") or "").strip()

if YANDEX_API_KEY:
    if not YANDEX_FOLDER_ID:
        raise SystemExit("Для Yandex AI Studio нужно задать YANDEX_FOLDER_ID.")
    MODEL = (os.getenv("YANDEX_MODEL") or
             f"gpt://{YANDEX_FOLDER_ID}/deepseek-v4.1-flash").strip()
    BASE_URL = "https://ai.api.cloud.yandex.net/v1"
    API_KEY = YANDEX_API_KEY
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        project=YANDEX_FOLDER_ID,
        timeout=LLM_TIMEOUT,
        max_retries=1,
    )
    PROVIDER = "Yandex AI Studio"
else:
    # Совместимые API (OpenAI, OpenRouter и др.) и локальная Ollama.
    MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not API_KEY and BASE_URL:
        API_KEY = "ollama"
    if not API_KEY:
        raise SystemExit(
            "Не заданы YANDEX_API_KEY/YANDEX_FOLDER_ID или OPENAI_API_KEY. "
            "Скопируйте .env.example в .env и заполните настройки."
        )
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=LLM_TIMEOUT,
        max_retries=1,
    )
    PROVIDER = "OpenAI-compatible/local"

print(
    f"[boot] env={_ENV_FILE.name if _ENV_FILE else 'переменные окружения'} | "
    f"provider={PROVIDER} | model={MODEL} | timeout={LLM_TIMEOUT:g}s | "
    f"llm_slots={LLM_MAX_CONCURRENT}",
    flush=True,
)
if BASE_URL and ("127.0.0.1" in BASE_URL or "localhost" in BASE_URL):
    print(
        "[boot] ВНИМАНИЕ: BASE_URL указывает на localhost. Это работает только "
        "если Ollama запущена на той же машине; на Railway такой адрес недоступен.",
        flush=True,
    )

# Railway: mount a persistent volume and set RPG_SAVES_DIR=/data/saves.
SAVES_DIR = Path(
    os.getenv("RPG_SAVES_DIR") or os.getenv("SAVES_DIR") or str(BASE_DIR / "saves")
).expanduser().resolve()
SAVES_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
try:
    SAVES_DIR.chmod(0o700)
except OSError:
    pass

# Registration and credentials are stored in a SQLite database in RPG_SAVES_DIR.
# Without a Railway Volume, *both accounts and saves* disappear on redeploy.
try:
    AUTH_USERS = json.loads(os.getenv("RPG_USERS_JSON", "{}"))
except json.JSONDecodeError as exc:
    raise RuntimeError("RPG_USERS_JSON должен содержать JSON-объект") from exc
if not isinstance(AUTH_USERS, dict) or any(
    not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", str(u)) or not isinstance(t, str) or not t
    for u, t in AUTH_USERS.items()
):
    raise RuntimeError("RPG_USERS_JSON: ожидается словарь username -> непустой токен")
if os.getenv("RPG_ACCESS_TOKEN"):
    AUTH_USERS["owner"] = os.environ["RPG_ACCESS_TOKEN"]
UNSAFE_LOCAL = os.getenv("RPG_ALLOW_UNSAFE_LOCAL", "").lower() in ("1", "true", "yes")
if not AUTH_USERS and not UNSAFE_LOCAL:
    raise RuntimeError("Для защиты API задайте RPG_ACCESS_TOKEN (или RPG_USERS_JSON).")

REGISTRATION_ENABLED = os.getenv("RPG_REGISTRATION_ENABLED", "1").lower() not in ("0", "false", "no")
REGISTRATION_CODE = os.getenv("RPG_REGISTRATION_CODE", "")
AI_CALLS_PER_HOUR = max(10, int(os.getenv("RPG_AI_CALLS_PER_HOUR", "180")))
AI_GLOBAL_CALLS_PER_HOUR = max(10, int(os.getenv("RPG_AI_GLOBAL_CALLS_PER_HOUR", "450")))
SESSION_TTL = 30 * 24 * 3600
DB_PATH = SAVES_DIR / "accounts.sqlite3"
USERNAME_RE = re.compile(r"[\w-]{3,32}$", re.UNICODE)
PASSWORD_MIN_LENGTH = 8


@contextmanager
def account_db():
    connection = sqlite3.connect(str(DB_PATH), timeout=15)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_account_db():
    with account_db() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, created_at INTEGER NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, expires_at INTEGER NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS rate_limits (
            bucket TEXT PRIMARY KEY, window_start INTEGER NOT NULL, hits INTEGER NOT NULL
        )""")


init_account_db()
try:
    DB_PATH.chmod(0o600)
except OSError:
    pass


def normalized_username(raw):
    if not isinstance(raw, str):
        return None
    name = unicodedata.normalize("NFKC", raw).strip().casefold()
    if not USERNAME_RE.fullmatch(name) or len(name) > 32:
        return None
    return name


def password_hash(password):
    # PBKDF2-HMAC with a fresh random salt; no plaintext passwords in SQLite.
    salt = secrets.token_bytes(16)
    rounds = 360_000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${salt.hex()}${derived.hex()}"


def verify_password(password, stored):
    try:
        algorithm, rounds, salt, expected = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256" or not 100_000 <= int(rounds) <= 1_000_000:
            return False
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(derived, bytes.fromhex(expected))
    except (AttributeError, TypeError, ValueError):
        return False


def use_quota(bucket, allowed, seconds):
    """Simple process-safe rate limiter shared by Gunicorn workers."""
    now = int(time.time())
    with account_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT window_start, hits FROM rate_limits WHERE bucket=?",
                         (bucket,)).fetchone()
        if not row or row["window_start"] + seconds <= now:
            db.execute("INSERT INTO rate_limits (bucket, window_start, hits) VALUES (?,?,1) "
                       "ON CONFLICT(bucket) DO UPDATE SET window_start=excluded.window_start, hits=1",
                       (bucket, now))
            return True
        if row["hits"] >= allowed:
            return False
        db.execute("UPDATE rate_limits SET hits=hits+1 WHERE bucket=?", (bucket,))
        return True


def session_token_for(username):
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with account_db() as db:
        db.execute("DELETE FROM sessions WHERE expires_at<=?", (now,))
        db.execute("INSERT INTO sessions(token_hash, username, expires_at) VALUES (?,?,?)",
                   (hashlib.sha256(token.encode()).hexdigest(), username, now + SESSION_TTL))
    return token


def user_saves_dir():
    folder = SAVES_DIR / g.current_user
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    return folder


@app.before_request
def require_api_token():
    public = {"/api/health", "/api/auth/login", "/api/auth/register", "/api/auth/config"}
    if not request.path.startswith("/api/") or request.path in public:
        return None
    if not AUTH_USERS and UNSAFE_LOCAL:
        if request.remote_addr not in ("127.0.0.1", "::1"):
            return jsonify({"error": "Локальный режим доступен только с этого компьютера."}), 403
        g.current_user = "owner"
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or len(auth) > 512:
        return jsonify({"error": "Требуется вход в аккаунт."}), 401
    token = auth[7:]
    token_digest = hashlib.sha256(token.encode()).hexdigest()
    with account_db() as db:
        row = db.execute("SELECT username FROM sessions WHERE token_hash=? AND expires_at>?",
                         (token_digest, int(time.time()))).fetchone()
    if row:
        g.current_user = row["username"]
    else:
        # Compatibility for older clients using manually configured owner keys.
        username = request.headers.get("X-RPG-User", "owner")
        expected = AUTH_USERS.get(username)
        if not expected or not hmac.compare_digest(token, expected):
            return jsonify({"error": "Сеанс истёк. Войдите заново."}), 401
        g.current_user = username
    if request.path in ("/api/chat", "/api/state", "/api/memory", "/api/diag"):
        if not use_quota("ai:user:" + g.current_user, AI_CALLS_PER_HOUR, 3600):
            return jsonify({"error": "Достигнут личный лимит ИИ-запросов за час. Попробуйте позже."}), 429
        if not use_quota("ai:global", AI_GLOBAL_CALLS_PER_HOUR, 3600):
            return jsonify({"error": "Общий лимит ИИ-запросов за час исчерпан. Попробуйте позже."}), 429
    return None


@app.route("/api/auth/config")
def api_auth_config():
    return jsonify({"registration_enabled": REGISTRATION_ENABLED,
                    "invite_required": bool(REGISTRATION_CODE)})


@app.route("/api/auth/register", methods=["POST"])
def api_auth_register():
    if not REGISTRATION_ENABLED:
        return jsonify({"error": "Регистрация временно закрыта."}), 403
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Укажите логин и пароль."}), 400
    username = normalized_username(data.get("username"))
    password = data.get("password")
    if not username:
        return jsonify({"error": "Логин: 3–32 символа, буквы, цифры, _ или -."}), 400
    if username in ("owner", "admin", "root") or any(
        username == u.casefold() for u in AUTH_USERS
    ):
        return jsonify({"error": "Это имя уже занято."}), 409
    if not isinstance(password, str) or not PASSWORD_MIN_LENGTH <= len(password) <= 128:
        return jsonify({"error": "Пароль должен содержать от 8 до 128 символов."}), 400
    if REGISTRATION_CODE and not hmac.compare_digest(str(data.get("invite_code", "")), REGISTRATION_CODE):
        return jsonify({"error": "Неверный код приглашения."}), 403
    if not use_quota("register:" + str(request.remote_addr), 8, 3600):
        return jsonify({"error": "Слишком много регистраций. Попробуйте позже."}), 429
    hashed = password_hash(password)
    with account_db() as db:
        try:
            db.execute("INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
                       (username, hashed, int(time.time())))
        except sqlite3.IntegrityError:
            return jsonify({"error": "Это имя уже занято."}), 409
    return jsonify({"ok": True, "user": username, "token": session_token_for(username)}), 201


@app.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Укажите логин и пароль."}), 400
    username = normalized_username(data.get("username"))
    password = data.get("password")
    if not username or not isinstance(password, str) or len(password) > 128:
        return jsonify({"error": "Неверный логин или пароль."}), 401
    if not use_quota("login:" + str(request.remote_addr) + ":" + username, 12, 900):
        return jsonify({"error": "Слишком много попыток. Повторите вход через 15 минут."}), 429
    with account_db() as db:
        account = db.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
    if account:
        valid = verify_password(password, account["password_hash"])
    else:
        # Optional legacy Railway credentials: owner and RPG_USERS_JSON.
        legacy_username = next((u for u in AUTH_USERS if u.casefold() == username), None)
        expected = AUTH_USERS.get(legacy_username) if legacy_username else None
        valid = bool(expected and hmac.compare_digest(password, expected))
        if valid:
            username = legacy_username
    if not valid:
        return jsonify({"error": "Неверный логин или пароль."}), 401
    return jsonify({"ok": True, "user": username, "token": session_token_for(username)})


@app.route("/api/auth/check")
def api_auth_check():
    return jsonify({"ok": True, "user": g.current_user})


@app.route("/api/auth/logout", methods=["POST"])
def api_auth_logout():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        digest = hashlib.sha256(auth[7:].encode()).hexdigest()
        with account_db() as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (digest,))
    return jsonify({"ok": True})


# Owner-only legacy migration. Never overwrite an already-migrated save file.
def migrate_legacy_saves():
    """Переносит старые общие и device-scoped сохранения в аккаунт owner."""
    folder = SAVES_DIR / "owner"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        folder.chmod(0o700)
    except OSError:
        pass
    for old_file in sorted(SAVES_DIR.glob("*.json")):
        # В старой версии имя имело вид <client-id>~<save-name>.json.
        # Тильда не разрешена в новом API-имени, поэтому заменяем её подчёркиванием.
        migrated_name = old_file.name.replace("~", "_")
        target = folder / migrated_name
        if target.exists():
            target = folder / ("legacy_" + migrated_name)
        if not target.exists():
            old_file.replace(target)


migrate_legacy_saves()

# Backward-compatible aliases from both source archives.
STATE_MODEL = (os.getenv("RPG_STATE_MODEL") or os.getenv("EXTRACT_MODEL") or MODEL).strip()
SUMMARY_MODEL = (os.getenv("RPG_SUMMARY_MODEL") or os.getenv("SUMMARY_MODEL") or MODEL).strip()
FREQ_PENALTY = float(os.getenv("OPENAI_FREQ_PENALTY", "0.35"))
PRES_PENALTY = float(os.getenv("OPENAI_PRES_PENALTY", "0.25"))
# Upper token budget for a single narrated reply. Not applied to state or memory models.
CHAT_MAX_TOKENS = max(256, min(1600, int(os.getenv("RPG_CHAT_MAX_TOKENS", "700"))))


@contextmanager
def llm_slot(timeout=None):
    """Ограничивает параллельные обращения к модели в одном процессе."""
    acquired = LLM_SEM.acquire(timeout=LLM_QUEUE_TIMEOUT if timeout is None else timeout)
    try:
        yield acquired
    finally:
        if acquired:
            LLM_SEM.release()


PENALTY_REJECT = (
    "frequency_penalty", "presence_penalty", "unrecognized", "unsupported",
    "unknown field", "unknown parameter", "extra fields", "additional properties",
    "unexpected keyword", "not permitted", "invalid_request",
)


def _short_error(exc):
    """Короткая диагностическая ошибка без вывода секретов окружения."""
    message = " ".join(str(exc).split())[:400]
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if body:
        try:
            message += " " + json.dumps(body, ensure_ascii=False)[:250]
        except (TypeError, ValueError):
            pass
    return f"{type(exc).__name__}{' ' + str(status) if status else ''}: {message}"


def _error_hint(exc):
    low = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status == 401 or "unauthorized" in low or "invalid api key" in low:
        return "Проверьте OPENAI_API_KEY или YANDEX_API_KEY."
    if status == 402 or "insufficient" in low or "credit" in low or "quota" in low:
        return "У провайдера закончились средства или квота."
    if status == 429 or "rate limit" in low:
        return "Провайдер ограничивает частоту запросов; попробуйте позже."
    if status == 404 or ("model" in low and "not found" in low):
        return "Модель не найдена; проверьте имя модели и доступ к ней."
    if ("connection" in low or "refused" in low or "timed out" in low
            or "name or service" in low):
        return "Нет связи с API. Проверьте OPENAI_BASE_URL и доступность провайдера."
    if "context" in low and ("length" in low or "token" in low):
        return "Контекст слишком длинный; начните новую партию или уменьшите историю."
    return ""


def _create_with_fallback(params):
    """Повторяет запрос без необязательных параметров, если провайдер их не поддерживает."""
    options = dict(params)
    last_error = None
    for _ in range(3):
        try:
            return client.chat.completions.create(**options)
        except Exception as exc:
            last_error = exc
            low = str(exc).lower()
            if "max_tokens" in low and "max_tokens" in options:
                options.pop("max_tokens", None)
                app.logger.warning("Провайдер отклонил max_tokens; повтор без лимита")
            elif any(token in low for token in PENALTY_REJECT) and (
                "frequency_penalty" in options or "presence_penalty" in options
            ):
                options.pop("frequency_penalty", None)
                options.pop("presence_penalty", None)
                app.logger.warning("Провайдер отклонил penalty-параметры; повтор без них")
            else:
                raise
    raise last_error


def _start_stream(messages):
    params = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.85,
        "max_tokens": CHAT_MAX_TOKENS,
        "stream": True,
        "frequency_penalty": FREQ_PENALTY,
        "presence_penalty": PRES_PENALTY,
    }
    return _create_with_fallback(params)


SYSTEM_PROMPT = """
Ты — ведущий (мастер) текстовой ролевой игры в жанре тёмного фэнтези / D&D / LitRPG.
Ты одновременно: рассказчик, шоураннер мира и арбитр правил.

════════ ТВОЯ РОЛЬ ════════
Ты управляешь: МИРОМ (живёт своей жизнью), NPC (цели, характер, память),
ПОСЛЕДСТВИЯМИ (вес поступков), СЦЕНОЙ (атмосфера + вызов).

Ты НЕ решаешь за игрока, что он делает, думает или говорит.

════════ СТИЛЬ И КРАТКОСТЬ — ОБЯЗАТЕЛЬНО ════════
• На каждую игровую реплику дай СТРОГО 4–6 КОРОТКИХ абзацев. Предпочитай 4; 5–6 только для сложных сцен. Отделяй абзацы одной пустой строкой.
• Каждый абзац — 1–3 коротких предложения. Ориентир: 90–170 слов на весь ответ, в кульминации не более 200. Не превращай каждый абзац в отдельную длинную сцену.
• Первый абзац сразу показывает результат действия игрока, без вступления и пересказа его слов.
• В остальных покажи одно значимое последствие, краткую реакцию NPC (если уместно) и естественное продолжение сцены.
• Атмосфера через одну-две точные детали: звук шагов по мокрому камню, запах дыма, дрожь света. Детали должны менять восприятие сцены, а не заполнять объём.
• Убирай тавтологию, внутренние монологи, философские рассуждения, чрезмерную драматизацию и повтор объяснённого ранее.
• Не вставляй заголовки, списки вариантов, таблицы, пересказ квестов, пересчёт характеристик или длинные справки в художественный ответ: для этого есть лист персонажа.
• Конкретика важнее эпитетов. Не «мрачный лес», а «сосны стоят так плотно, что снег не падает на землю».
• Не решай за игрока и не говори, что он думает; оставляй пространство для его собственного действия.
• Если игрок прямо попросил полные правила или текст документа, ответь по существу, без лишнего художественного наполнения.

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
«+<N> XP» отдельной короткой строкой ВНУТРИ последнего абзаца, без пустой строки перед ней: это не новый абзац.
Пример: «Гоблин падает. Из тоннеля доносится второй топот». Следующей строкой: «+50 XP».
Если бой был долгим, можно в конце: «+150 XP за троих гоблинов».
Если игрок перехитрил без боя, можно в конце: «+25 XP за смекалку».

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

Не выкладывай в повествовании весь лист персонажа и длинные списки характеристик: они доступны в профиле и отслеживаются модулем состояния.
Объяви героя одной короткой фразой («Кейр — эльф-плут с прошлым преступника») и переходи к первой сцене.
Если для выбора игрока критичны конкретная характеристика или умение, назови только их.

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
СОСТОЯНИЕ: темнота +3, слепота +5, ранение +2, крит. +3, отравлен +2, испуг +2, спешка +2, местность +1..+2.
ПОМОЩЬ: союзник −2, подготовка −2..−3, знакомое −1..−2.

СВЯЗАННЫЕ ДЕЙСТВИЯ:
• «Лежит, хочет атаковать» — сначала подъём DC 10, потом атака.
• «Обездвижен» — Сила DC 15.
• «Связан» — Ловкость рук DC 15 или Сила DC 20.

КАК ОБЪЯВЛЯТЬ:
«Сделайте бросок 1d20+<Мод> (<Хар>) +<Бонус> (<Навык>) (Сложность <X>: база <Y> + <причины>)».

Когда приходит сообщение [[БРОСОК]], число после «=» — уже КОНЕЧНЫЙ результат со всеми указанными бонусами и штрафами. Не прибавляй модификаторы к нему ещё раз. Сравни готовый итог с объявленной DC; натуральные результаты кубика указаны в квадратных скобках.

Результаты: 1 — крит. провал, 2–9 — провал с ценой, 10–14 — успех с осложнением, 15–19 — чистый успех, 20 — крит. успех.

════════ СТРУКТУРА ОТВЕТА — 4–6 АБЗАЦЕВ ════════
Абзац 1: непосредственный результат действия игрока.
Абзац 2: одна выразительная деталь места или последствие действия.
Абзац 3: краткая реплика/реакция NPC, если есть; иначе — движение сцены.
Абзац 4: новая ситуация, напряжение или доступный выбор БЕЗ шаблонного вопроса.
Абзацы 5–6 добавляй ТОЛЬКО если важное событие иначе не поместилось (например, бой или поворот сюжета).
Не размечай эти пункты номерами или заголовками в готовом ответе.
В ответе должно быть НЕ МЕНЬШЕ 4 и НЕ БОЛЬШЕ 6 абзацев, в том числе при создании героя.
Если начисляется XP, добавь строку «+N XP» к ПОСЛЕДНЕМУ абзацу без пустой строки и не начинай отдельный седьмой абзац.
Последний сюжетный момент оставляй открытым для инициативы игрока; не навязывай ему поступок.

════════ NPC ════════
У значимого NPC: имя, роль, яркая деталь, цель, тайна, манера речи.
Помнят поступки, могут лгать, не выкладывают всё сразу.

════════ КАНОН ════════
Запоминай всё: имя/расу/класс/предысторию, характеристики, инвентарь, заметки, квесты, позицию, NPC, места.

════════ ЯЗЫК ════════
Русский, естественный. Жёсткость — по жанру.

════════ СТАРТ ════════
Первый ответ — после описания персонажа/мира. Сначала раса/класс/предыстория ОДНОЙ фразой, потом сразу сцена.
Даже в первом ответе — только 4–6 коротких абзацев. Не перечисляй весь лист персонажа.
"""


STATE_EXTRACT_PROMPT = """
Ты — модуль отслеживания состояния персонажа в RPG.
Получив текущее состояние и последние реплики, верни ТОЛЬКО ИЗМЕНЕНИЯ
в виде JSON-объекта. Без markdown. Если ничего не изменилось, верни {}.
Скалярные поля указывай только при явном изменении. Неизменённые поля НЕ ВОЗВРАЩАЙ.
Если изменён массив, верни его ПОЛНУЮ актуальную версию, включая [] при удалении
всех элементов. Отсутствующий ключ означает «оставить как есть».
Не выдумывай изменений и не восстанавливай выброшенные предметы.

Возможные поля и структура (это схема, НЕ копируй весь объект):
{
  "name": "имя или Пустой",
  "race": "", "class": "", "background": "",
  "level": 1, "xp": 0, "xp_next": 100,
  "hp": 10, "max_hp": 10, "gold": 0,
  "stats": {"str":10,"dex":10,"con":10,"int":10,"wis":10,"cha":10},
  "proficiencies": [], "skills": [], "abilities": [],
  "inventory": [], "notes": [], "quests": [], "conditions": [],
  "relationships": [], "faction_clocks": [],
  "dc_modifier": 0, "dc_modifier_reasons": [],
  "position": "стоит", "cover": "нет",
  "weapon_in_hand": "", "restrained": false,
  "level_history": [], "xp_event_ids": []
}

XP, LEVEL, XP_NEXT, LEVEL_PENDING — НЕ ИЗМЕНЯЙ самостоятельно. Не возвращай их в изменениях.
XP начисляется сервером только из новых реплик мастера с явным форматом «+N XP».
Не рассчитывай награды и не повышай уровень; сервер обработает это отдельно.

⚠️ INVENTORY vs NOTES:
- inventory — ТОЛЬКО физические предметы: оружие, броня, зелья, еда, инструменты, монеты, украшения, ключи.
- notes — всё, что МОЖНО ПРОЧИТАТЬ: письма, документы, заметки, дневники, уведомления, слухи, грамоты, карты (как информация).
- Если мастер пишет «ты находишь письмо», «находишь дневник», «тебе вручают грамоту», «системное уведомление», «слышишь слух» — это в notes, НЕ в inventory.
- Ключ от сундука — inventory. Записка с кодом — notes.

СТРУКТУРА NOTES:
Массив объектов:
{
  "id": "n1",
  "type": "note|letter|document|notification|rumor|journal",
  "title": "Краткое название (до 40 символов)",
  "content": "Полный текст или суть (1–4 предложения)",
  "source": "Кто/откуда (имя NPC, место)",
  "date": "Игровая дата или 'недавно'",
  "unread": true
}
- type: note (своя заметка), letter (письмо), document (документ), notification (системное), rumor (слух), journal (дневник/журнал).
- unread: true для новых, false если игрок прочитал.
- Не дублируй. Если заметка уже есть — не добавляй повторно.
- Максимум 30 заметок. Старые прочитанные можно удалять (оставляй важные).

ОСНОВНЫЕ ПОЛЯ:
- Не выдумывай событий.
- HP не превышает max_hp.
- Инвентарь — без дублей.

RACE, CLASS, BACKGROUND: сохрани, если уже есть. Меняй только при событии в игре.

STATS: если раса/класс только появились и все 10 — выставь по классу + расовые бонусы. Если уже не 10 — не пересчитывай.

PROFICIENCIES: массив строк. Формируется ОДИН РАЗ. Максимум 12.

SKILLS: массив {"name":"Скрытность","bonus":5,"source":"Плут"}. ОДИН РАЗ, 6–8 навыков.

ABILITIES: массив {"name":"Ярость","description":"...","uses_max":2,"uses_left":2,"recharge":"long_rest"}.
recharge: passive (uses_max=0), short_rest, long_rest, day.
uses_left уменьшай при использовании, восстанавливай при отдыхе.
Назначай ОДИН РАЗ, 5–6 умений.

DC_MODIFIER: только реальные причины. Нейтрально → 0, [].

POSITION: "стоит", "сидит", "на коленях", "лежит", "верхом", "плавает", "висит", "схвачен", "обездвижен".
COVER: "нет", "половинное", "три_четверти", "полное".
WEAPON_IN_HAND: "", "кинжал", "меч", "лук", "посох", "щит", "обе_руки_заняты".
RESTRAINED: bool.

QUESTS: массив объектов с полями id/name/type/status/chain/chain_order/stages/current_stage/rewards/giver/location.
- Все stages done → status "completed", XP выдаётся в этом же ходу.
- Если chain и completed → сразу создай следующий квест (chain_order + 1).
- Не продлевай завершённый. Новый этап = новый квест.
- Максимум 8 активных.

LEVEL_HISTORY: массив {"level":2,"hp":"+4","stats":"+1 Сила","abilities":"+Рывок"}.

ПРОКАЧКА:
- xp >= xp_next → level+1, max_hp += 2+мод Тел, hp=max_hp, stats: одна хар +1, abilities: новое умение, skills: пересчёт при 5/9/13/17, xp_next: см. прогрессию.
- В level_history — запись.
- Max 1 уровень за ход.

RELATIONSHIPS: {"name":"Торвин","attitude":"насторожен","note":"помог"}. Максимум 6–8.

FACTION_CLOCKS: {"name":"Культ","progress":3,"max":6,"goal":"..."}. 1–4 силы.
"""


SUMMARY_PROMPT = """
Ты — архивариус. Сожми фрагмент диалога в краткое резюме.

СОХРАНИ:
- Ключевые события, имена NPC, отношение к игроку
- Решения и последствия
- Открытые вопросы, цели, долги
- Факты мира
- Расу, класс, предысторию, характеристики
- Текущий XP, уровень, активные квесты и их этапы
- Активные заметки (письма, документы — их суть)
- Текущее положение

НЕ ВКЛЮЧАЙ: дословные реплики, эмоции, повторы.

Формат: компактное резюме до 2000 символов. Включи ключевые имена и нерешённые последствия, без повторов.
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
    "level_history": [], "xp_event_ids": [], "level_pending": 0
}


# Memory is stored in the client's local session AND in the save file.
# Each new summary incorporates only dialogue that has not been summarized before.
MEMORY_UPDATE_AFTER = max(6, int(os.getenv("RPG_MEMORY_UPDATE_AFTER", "16")))
KEEP_RECENT = 10
MAX_DIALOGUE_CHARS = 18000


def dialogue_messages(raw):
    """Ignore client-supplied system prompts; cap per-message text for model requests."""
    if not isinstance(raw, list):
        return []
    return [{"role": m["role"], "content": m["content"][:MAX_DIALOGUE_CHARS]}
            for m in raw if isinstance(m, dict) and
            m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]


def prefix_digest(dialogue, count):
    data = json.dumps(dialogue[:count], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def verified_memory(dialogue, memory):
    """Prevent stale summaries from leaking into a reset or edited game."""
    if not isinstance(memory, dict):
        return {"summary": "", "covered_count": 0, "prefix_hash": ""}
    summary = memory.get("summary", "")
    count = memory.get("covered_count", 0)
    digest = memory.get("prefix_hash", "")
    if not isinstance(summary, str) or not isinstance(count, int) or count < 0 or count > len(dialogue):
        return {"summary": "", "covered_count": 0, "prefix_hash": ""}
    if count and (not digest or digest != prefix_digest(dialogue, count)):
        return {"summary": "", "covered_count": 0, "prefix_hash": ""}
    if not summary.strip():
        return {"summary": "", "covered_count": 0, "prefix_hash": ""}
    return {"summary": summary[:6000], "covered_count": count, "prefix_hash": digest}


def build_chat_messages(raw_messages, raw_state, raw_memory):
    dialogue = dialogue_messages(raw_messages)
    memory = verified_memory(dialogue, raw_memory)
    state = raw_state if isinstance(raw_state, dict) else DEFAULT_STATE
    current_state = {k: v for k, v in state.items() if k not in ("xp_event_ids", "level_history")}
    context = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "system", "content": "[АКТУАЛЬНОЕ СОСТОЯНИЕ ПЕРСОНАЖА]\n" +
                json.dumps(current_state, ensure_ascii=False)}]
    if memory["summary"]:
        context.append({"role": "system", "content": "[РЕЗЮМЕ ПРОШЛОГО]\n" + memory["summary"]})
    # For old saves without memory the full dialogue is used once; the frontend
    # builds a persisted summary as soon as it restores a legacy campaign.
    return context + dialogue[memory["covered_count"]:]


def update_campaign_memory(raw_messages, raw_memory):
    dialogue = dialogue_messages(raw_messages)
    memory = verified_memory(dialogue, raw_memory)
    cutoff = len(dialogue) - KEEP_RECENT
    if cutoff <= memory["covered_count"] or cutoff - memory["covered_count"] < MEMORY_UPDATE_AFTER:
        return memory
    new_chunk = dialogue[memory["covered_count"]:cutoff]
    new_text = "\n".join(f"{m['role']}: {m['content']}" for m in new_chunk)
    if not new_text.strip():
        return memory
    archive = ("ПРЕДЫДУЩАЯ ПАМЯТЬ:\n" + memory["summary"] + "\n\n" if memory["summary"] else "")
    started = time.monotonic()
    try:
        with llm_slot() as acquired:
            if not acquired:
                raise RuntimeError("очередь запросов к ИИ переполнена")
            result = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "system", "content": SUMMARY_PROMPT},
                          {"role": "user", "content": archive + "НОВЫЕ СОБЫТИЯ:\n" + new_text}],
                temperature=0.2,
            )
        summary = (result.choices[0].message.content or "").strip()
        if not summary:
            return memory
        usage = getattr(result, "usage", None)
        app.logger.info("memory seconds=%.2f tokens=%s new_messages=%d",
                        time.monotonic() - started,
                        getattr(usage, "total_tokens", "n/a"), len(new_chunk))
        return {"summary": summary[:6000], "covered_count": cutoff,
                "prefix_hash": prefix_digest(dialogue, cutoff)}
    except Exception:
        app.logger.exception("Failed to update campaign memory")
        return memory


def roll_dice(expr):
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
        # Проверка минимальных полей
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


def extract_state(messages, current_state):
    current = dict(DEFAULT_STATE)
    if isinstance(current_state, dict):
        current.update(current_state)
    current.setdefault("xp_event_ids", [])
    if not isinstance(current.get("xp_event_ids"), list):
        current["xp_event_ids"] = []

    # On legacy saves, mark historical XP-bearing lines as already seen, except
    # the latest assistant message (which may be the just-finished turn).
    if "xp_event_ids" not in (current_state or {}):
        for index, m in enumerate(messages):
            if m.get("role") == "assistant" and XP_EVENT_RE.search(m.get("content", "")) and index != (len(messages) - 1):
                current["xp_event_ids"].append(message_event_id(m, index))

    fresh_events = []
    for index, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        event_id = message_event_id(m, index)
        if event_id in current["xp_event_ids"]:
            continue
        matches = [int(x) for x in XP_EVENT_RE.findall(content)]
        if matches:
            fresh_events.append((event_id, sum(matches)))

    recent = dialogue_messages(messages)[-4:]
    dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    # XP event IDs are only for server-side deduplication and waste model tokens.
    model_state = {k: v for k, v in current.items() if k != "xp_event_ids"}
    prompt = (f"ТЕКУЩЕЕ СОСТОЯНИЕ:\n{json.dumps(model_state, ensure_ascii=False)}\n\n"
              f"ПОСЛЕДНИЕ РЕПЛИКИ:\n{dialogue}\n\nВерни только изменённые поля JSON."
              " XP, уровень, xp_next, level_pending и xp_event_ids НЕ ВОЗВРАЩАЙ.")
    started = time.monotonic()
    try:
        with llm_slot() as acquired:
            if not acquired:
                raise RuntimeError("очередь запросов к ИИ переполнена")
            resp = client.chat.completions.create(
                model=STATE_MODEL,
                messages=[{"role": "system", "content": STATE_EXTRACT_PROMPT},
                         {"role": "user", "content": prompt}],
                temperature=0.1,
            )
        new_state = clean_json(resp.choices[0].message.content or "")
        usage = getattr(resp, "usage", None)
        app.logger.info("state seconds=%.2f tokens=%s fields=%s",
                        time.monotonic() - started,
                        getattr(usage, "total_tokens", "n/a"),
                        ",".join(new_state.keys()) if isinstance(new_state, dict) else "bad-json")
    except Exception:
        app.logger.exception("State extraction failed")
        new_state = {}
    if not isinstance(new_state, dict):
        new_state = {}
    # These fields are always calculated by the server, never by the model.
    for protected in ("xp", "xp_next", "level", "level_pending", "xp_event_ids"):
        new_state.pop(protected, None)

    merged = dict(current)
    # A present array is authoritative, including [] (drop/discard/completed removal).
    # Missing arrays and fields remain unchanged. This avoids the old "immortal items" bug.
    for key, value in new_state.items():
        if key not in DEFAULT_STATE or value is None:
            continue
        if isinstance(DEFAULT_STATE[key], list) and not isinstance(value, list):
            continue
        if isinstance(DEFAULT_STATE[key], dict) and not isinstance(value, dict):
            continue
        if key == "stats" and isinstance(value, dict):
            merged[key] = {**(current.get("stats") or {}),
                           **{k: v for k, v in value.items() if k in DEFAULT_STATE["stats"]}}
        else:
            merged[key] = value
    # Server-authoritative XP: only explicit new rewards, never double-counted.
    xp = max(0, int(current.get("xp", 0) or 0))
    processed = list(current.get("xp_event_ids", []))
    for event_id, amount in fresh_events:
        if event_id not in processed:
            xp += amount
            processed.append(event_id)
    merged["xp"] = xp
    merged["xp_event_ids"] = processed[-5000:]
    # Protect core scalar values from blank/null model output.
    for f in ("name", "race", "class", "background", "position", "cover", "weapon_in_hand"):
        if not merged.get(f) and current.get(f):
            merged[f] = current[f]
    def as_int(value, default=0):
        try:
            return int(value)
        except (ValueError, TypeError, OverflowError):
            return default
    merged["max_hp"] = max(1, as_int(merged.get("max_hp"), 10))
    merged["hp"] = max(0, min(as_int(merged.get("hp")), merged["max_hp"]))
    previous_level = max(1, int(current.get("level", 1) or 1))
    derived_level = level_for_xp(merged["xp"])
    merged["level"] = derived_level
    merged["xp_next"] = xp_threshold_for_level(derived_level)
    merged["level_pending"] = max(0, derived_level - previous_level)
    merged["gold"] = max(0, as_int(merged.get("gold")))
    merged["dc_modifier"] = as_int(merged.get("dc_modifier"))
    if not isinstance(merged.get("name"), str):
        merged["name"] = current.get("name", DEFAULT_STATE["name"])
    stats = merged.get("stats")
    if not isinstance(stats, dict):
        stats = current.get("stats", DEFAULT_STATE["stats"])
    merged["stats"] = {key: as_int(stats.get(key), 10) for key in DEFAULT_STATE["stats"]}
    for f in ("race", "class", "background", "position", "cover", "weapon_in_hand"):
        if not isinstance(merged.get(f), str): merged[f] = current.get(f, DEFAULT_STATE[f])
    if not isinstance(merged.get("restrained"), bool): merged["restrained"] = bool(current.get("restrained", False))
    for f in ("inventory", "notes", "quests", "conditions", "relationships", "faction_clocks",
              "dc_modifier_reasons", "proficiencies", "skills", "abilities", "level_history", "xp_event_ids"):
        if not isinstance(merged.get(f), list): merged[f] = current.get(f, [])
    merged["quests"] = [q for q in (normalize_quest(x) for x in merged["quests"]) if q]
    merged["notes"] = [n for n in (normalize_note(x) for x in merged["notes"]) if n]
    return merged


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "model": MODEL,
        "state_model": STATE_MODEL,
        "summary_model": SUMMARY_MODEL,
        "base_url_configured": bool(BASE_URL),
        "provider": PROVIDER,
        "llm_max_concurrent": LLM_MAX_CONCURRENT,
    })


@app.route("/api/diag")
def api_diag():
    """Настройки и опциональная проверка реального ответа провайдера."""
    result = {
        "provider": PROVIDER,
        "model": MODEL,
        "state_model": STATE_MODEL,
        "summary_model": SUMMARY_MODEL,
        "base_url_configured": bool(BASE_URL),
        "key_set": bool(API_KEY),
        "env_file": _ENV_FILE.name if _ENV_FILE else None,
        "llm_timeout": LLM_TIMEOUT,
        "llm_max_concurrent": LLM_MAX_CONCURRENT,
        "llm_queue_timeout": LLM_QUEUE_TIMEOUT,
        "saves_dir": str(SAVES_DIR),
    }
    args = getattr(request, "args", {})
    if args.get("ping") == "1":
        started = time.monotonic()
        try:
            with llm_slot(timeout=20) as acquired:
                if not acquired:
                    raise RuntimeError("нет свободного слота модели")
                response = _create_with_fallback({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Ответь одним словом: ок"}],
                    "temperature": 0,
                    "max_tokens": 8,
                })
            result["ping_ok"] = True
            result["ping"] = (response.choices[0].message.content or "").strip()
            result["ping_seconds"] = round(time.monotonic() - started, 2)
        except Exception as exc:
            result["ping_ok"] = False
            result["ping_error"] = _short_error(exc)
            hint = _error_hint(exc)
            if hint:
                result["ping_hint"] = hint
    return jsonify(result)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/roll", methods=["POST"])
def api_roll():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(roll_dice(data.get("expr", "")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    raw_messages = data.get("messages") or []
    if not dialogue_messages(raw_messages):
        return jsonify({"error": "empty messages"}), 400
    messages = build_chat_messages(raw_messages, data.get("state"), data.get("memory"))

    def generate():
        started = time.monotonic()
        char_count = 0
        try:
            with llm_slot() as acquired:
                if not acquired:
                    yield "\n[Ошибка API: сервер занят — все слоты модели заняты. Подождите и повторите.]"
                    return
                stream = _start_stream(messages)
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    piece = getattr(delta, "content", None) if delta else None
                    if piece:
                        char_count += len(piece)
                        yield piece
        except Exception as exc:
            app.logger.exception("Chat completion failed: %s", _short_error(exc))
            hint = _error_hint(exc)
            yield "\n[Ошибка API: " + _short_error(exc) + ((" " + hint) if hint else "") + "]"
        finally:
            app.logger.info("chat seconds=%.2f chars=%d context_messages=%d",
                            time.monotonic() - started, char_count, len(messages))

    return Response(
        generate(),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/memory", methods=["POST"])
def api_memory():
    data = request.get_json(silent=True) or {}
    return jsonify(update_campaign_memory(data.get("messages"), data.get("memory")))


@app.route("/api/state", methods=["POST"])
def api_state():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    state = data.get("state") or DEFAULT_STATE
    if not isinstance(messages, list) or not messages:
        return jsonify(state)
    return jsonify(extract_state(messages, state))


@app.route("/api/saves")
def api_saves_list():
    return jsonify(sorted(p.stem for p in user_saves_dir().glob("*.json") if not p.name.endswith(".bak.json")))


def safe_save_name(raw_name):
    # Preserve Cyrillic and other unicode letters, reject traversal and control chars.
    name = unicodedata.normalize("NFKC", str(raw_name or "")).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w-]", "", name, flags=re.UNICODE).strip("_- ")
    return name[:64]


@app.route("/api/save/rename", methods=["POST"])
def api_save_rename():
    data = request.get_json(silent=True) or {}
    old_name = safe_save_name(data.get("old_name"))
    new_name = safe_save_name(data.get("new_name"))
    if not old_name or not new_name:
        return jsonify({"error": "Укажите имя сохранения из букв, цифр, _ или -."}), 400
    old_path = user_saves_dir() / f"{old_name}.json"
    new_path = user_saves_dir() / f"{new_name}.json"
    if not old_path.exists():
        return jsonify({"error": "Сохранение не найдено."}), 404
    if old_name == new_name:
        return jsonify({"ok": True, "name": new_name})
    if new_path.exists():
        return jsonify({"error": "Сохранение с таким именем уже существует."}), 409
    try:
        old_path.rename(new_path)
        old_backup = user_saves_dir() / f"{old_name}.bak.json"
        new_backup = user_saves_dir() / f"{new_name}.bak.json"
        if old_backup.exists() and not new_backup.exists():
            old_backup.rename(new_backup)
    except OSError as e:
        return jsonify({"error": f"Не удалось переименовать сохранение: {e}"}), 500
    return jsonify({"ok": True, "name": new_name})


@app.route("/api/save/delete", methods=["POST"])
def api_save_delete():
    data = request.get_json(silent=True) or {}
    name = safe_save_name(data.get("name"))
    if not name:
        return jsonify({"error": "Некорректное имя сохранения."}), 400
    path = user_saves_dir() / f"{name}.json"
    if not path.exists():
        return jsonify({"error": "Сохранение не найдено."}), 404
    try:
        path.unlink()
        backup = user_saves_dir() / f"{name}.bak.json"
        if backup.exists():
            backup.unlink()
    except OSError as e:
        return jsonify({"error": f"Не удалось удалить сохранение: {e}"}), 500
    return jsonify({"ok": True, "name": name})


def atomic_write_text(path, content):
    """Записывает JSON через временный файл и атомарную замену."""
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(silent=True) or {}
    name = safe_save_name(data.get("name")) or "default"
    folder = user_saves_dir()
    path = folder / f"{name}.json"

    # UI-only cards are stored separately from dialogue/XP history.
    ui_events = data.get("ui_events") or []
    if not isinstance(ui_events, list):
        ui_events = []
    messages = data.get("messages") or []
    if not isinstance(messages, list):
        return jsonify({"error": "messages должен быть списком"}), 400
    payload = {
        "schema_version": 4,
        "messages": messages,
        "state": data.get("state") if isinstance(data.get("state"), dict) else DEFAULT_STATE,
        "memory": data.get("memory") if isinstance(data.get("memory"), dict) else {},
        "ui_events": ui_events[:2000],
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    # Сначала атомарно сохраняем предыдущую версию в .bak.json.
    if path.exists():
        backup = folder / f"{name}.bak.json"
        try:
            atomic_write_text(backup, path.read_text(encoding="utf-8"))
        except OSError:
            app.logger.warning("Не удалось обновить резервную копию %s", backup.name)
    atomic_write_text(path, serialized)
    return jsonify({"ok": True, "name": name, "saved_at": path.stat().st_mtime})


@app.route("/api/load/<name>")
def api_load(name):
    name = safe_save_name(name)
    if not name:
        return jsonify({"error": "invalid save name"}), 400

    folder = user_saves_dir()
    primary = folder / f"{name}.json"
    backup = folder / f"{name}.bak.json"
    if not primary.exists() and not backup.exists():
        return jsonify({"error": "not found"}), 404

    errors = []
    for candidate in (primary, backup):
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(data, list):
                data = {"messages": data, "state": DEFAULT_STATE}
            if not isinstance(data, dict) or not isinstance(data.get("messages", []), list):
                raise ValueError("неверная структура сохранения")
            if candidate == backup:
                data["recovered_from_backup"] = True
            return jsonify(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{candidate.name}: {exc}")

    return jsonify({"error": "save file and backup are corrupted",
                    "details": errors[:2]}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1" if UNSAFE_LOCAL else "0.0.0.0", port=port, debug=False, threaded=True)
