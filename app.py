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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

from flask import Flask, request, jsonify, Response, render_template, stream_with_context
from itsdangerous import BadSignature, URLSafeSerializer
from openai import OpenAI

from campaign import (
    active_check, apply_ability_event, facts_context, formula_key, merge_facts, merge_scene, notes_context,
    initial_identity, normalize_abilities, normalize_ability, pending_check, roll_result, rules_result,
    sheet_identity,
)

from token_usage import TokenUsage, state_turn_id

BASE_DIR = Path(__file__).resolve().parent
APP_REVISION = "2026-11-26.3"
QWEN_SUMMARY_LIMIT = 480
QWEN_STATE_LIMIT = 1100
QWEN_STATE_SETUP_LIMIT = 2200


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

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))
YANDEX_API_KEY = (os.getenv("YANDEX_API_KEY") or "").strip()
YANDEX_FOLDER_ID = (os.getenv("YANDEX_FOLDER_ID") or "").strip()
USE_YANDEX = bool(YANDEX_API_KEY)
PROVIDER = "Yandex AI Studio" if USE_YANDEX else "OpenAI-compatible"
KEY_VARIABLE = "YANDEX_API_KEY" if USE_YANDEX else "OPENAI_API_KEY"
MODEL_VARIABLE = "YANDEX_MODEL" if USE_YANDEX else "OPENAI_MODEL"


def _model_id(value):
    if USE_YANDEX and not value.startswith("gpt://"):
        return f"gpt://{YANDEX_FOLDER_ID}/{value}"
    return value


client_options = {"timeout": LLM_TIMEOUT, "max_retries": 1}
if USE_YANDEX:
    if not YANDEX_FOLDER_ID:
        raise SystemExit("Для Yandex AI Studio задайте YANDEX_FOLDER_ID в Railway Variables.")
    BASE_URL = "https://ai.api.cloud.yandex.net/v1"
    API_KEY = YANDEX_API_KEY
    MODEL = _model_id((os.getenv("YANDEX_MODEL") or "").strip() or "deepseek-v4.1-flash")
    client_options["project"] = YANDEX_FOLDER_ID
else:
    BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
    MODEL = (os.getenv("OPENAI_MODEL") or "").strip() or "gpt-4o-mini"
    if not API_KEY and BASE_URL:
        API_KEY = "ollama"
    if not API_KEY:
        raise SystemExit(
            "В Railway Variables задайте YANDEX_API_KEY и YANDEX_FOLDER_ID для Yandex AI Studio "
            "или OPENAI_API_KEY для OpenAI."
        )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, **client_options)
ROLL_SIGNER = URLSafeSerializer(os.getenv("ROLL_SIGNING_KEY") or API_KEY, salt="rpg-dice-v1")
TURN_COMPLETE = "\n[[ХОД_ЗАВЕРШЁН]]"

# Без отдельных настроек служебные запросы используют основную модель.
EXTRACT_MODEL = _model_id((os.getenv("EXTRACT_MODEL") or "").strip() or MODEL)
SUMMARY_MODEL = _model_id((os.getenv("SUMMARY_MODEL") or "").strip() or MODEL)

# ─── Длина ответа мастера и бюджет хода ───────────────────────────────────
# Эти константы объявлены до первого print(): модуль выполняется сверху вниз,
# и обращение к ним выше определения роняло воркер с NameError.
# Ориентир: ~7000 токенов на обычный ход (ответ мастера + лист + свёртка).
# presence_penalty снят: он подталкивал модель к раннему завершению ответа.
FREQ_PENALTY = float(os.getenv("OPENAI_FREQ_PENALTY", "0.15"))
PRES_PENALTY = float(os.getenv("OPENAI_PRES_PENALTY", "0.0"))
NARRATION_MAX_TOKENS = max(256, int(os.getenv("NARRATION_MAX_TOKENS", "900")))
# Создание героя длиннее: блок листа плюс открывающая сцена. 2200, потому что
# обрыв на середине листа стоит целого хода.
NARRATION_SETUP_TOKENS = max(NARRATION_MAX_TOKENS, int(os.getenv("NARRATION_SETUP_TOKENS", "2200")))
# Повтор при пустом или оборванном ответе: лимит выше, рассуждения выключены.
NARRATION_RETRY_TOKENS = max(NARRATION_SETUP_TOKENS, int(os.getenv("NARRATION_RETRY_TOKENS", "2600")))
NARRATION_RETRY_EFFORT = (os.getenv("NARRATION_RETRY_EFFORT") or "none").strip()
# Необязательно: "none", "minimal" или "low" — если модель тратит вывод на рассуждения.
CHAT_REASONING_EFFORT = (os.getenv("CHAT_REASONING_EFFORT") or "").strip()

# Что выяснилось про параметр рассуждений у текущей модели (память процесса):
#   None — ещё не проверяли, "" — провайдер параметр не принимает,
#   иначе — режим, который сработал. Так повтор не оплачивается каждый ход.
_REASONING_STATE = {"fix": None}

print(
    f"[boot] app_revision={APP_REVISION} | env={_ENV_FILE.name if _ENV_FILE else 'нет (только переменные окружения)'} | "
    f"provider={PROVIDER} | model={MODEL} | base_url={BASE_URL or 'https://api.openai.com/v1'} | "
    f"timeout={LLM_TIMEOUT:g}s | narration_max_tokens={NARRATION_MAX_TOKENS} | "
    f"narration_setup_tokens={NARRATION_SETUP_TOKENS}",
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
TOKEN_USAGE = TokenUsage(SAVES_DIR / "token_usage.sqlite3")
app.add_url_rule("/admin/tokens", view_func=TOKEN_USAGE.dashboard)

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
        return f"Ключ не принят. Проверьте {KEY_VARIABLE} в Railway Variables."
    if status == 403 and USE_YANDEX:
        return "Доступ запрещён. Проверьте YANDEX_FOLDER_ID и права ключа Yandex AI Studio."
    if status == 402 or "insufficient" in low or "credit" in low or "quota" in low:
        return "У провайдера закончились средства. Пополните счёт или смените модель."
    if status == 429 or "rate limit" in low:
        return "Провайдер ограничивает частоту. Подождите минуту или смените модель."
    if status == 404 or ("model" in low and "not found" in low):
        return f"Модель не найдена. Проверьте {MODEL_VARIABLE} в Railway Variables."
    if ("connection" in low or "refused" in low or "timed out" in low
            or "name or service" in low):
        if USE_YANDEX:
            return "Нет связи с Yandex AI Studio. Проверьте соединение и логи Railway."
        return ("Нет связи с моделью. Проверьте OPENAI_BASE_URL: на Railway "
                "адрес 127.0.0.1 указывает на сам контейнер, где Ollama нет.")
    if "context" in low and ("length" in low or "token" in low):
        return "Слишком длинный контекст. Начните новую партию или уменьшите историю."
    return ""


def _is_yandex_qwen(model):
    host = urlsplit(BASE_URL or "").hostname
    yandex_endpoint = host in {
        "ai.api.cloud.yandex.net", "apps.ai.api.cloud.yandex.net", "llm.api.cloud.yandex.net",
    }
    return (USE_YANDEX or yandex_endpoint) and "qwen3.6-35b-a3b" in model.lower().split("/")


def _qwen_service_options(model, limit):
    if not _is_yandex_qwen(model):
        return {}
    # extra_body also works with versions of the OpenAI SDK that predate this limit.
    return {"extra_body": {"reasoning_effort": "none", "max_completion_tokens": limit}}


def _narration_options(messages, limit, stream, reasoning_effort):
    """Общие параметры рассказа: и для потока, и для готового ответа."""
    options = dict(model=MODEL, messages=messages, max_tokens=limit)
    if stream:
        options.update(stream=True, stream_options={"include_usage": True})
    if _is_yandex_qwen(MODEL):
        # Use Qwen's non-thinking sampling profile for narration.
        options.pop("max_tokens", None)
        options.update(temperature=0.7, top_p=0.8,
                       extra_body={"reasoning_effort": "none", "max_completion_tokens": limit})
    else:
        options.update(temperature=0.8, frequency_penalty=FREQ_PENALTY,
                       presence_penalty=PRES_PENALTY)
        if reasoning_effort:
            options["extra_body"] = {"reasoning_effort": reasoning_effort}
    return options


def _create_with_fallback(options, diag=None):
    """Запрос к модели с откатом по параметрам, которые провайдер не принял."""
    while True:
        try:
            return client.chat.completions.create(**options)
        except Exception as exc:
            detail = str(exc).lower()
            status = getattr(exc, "status_code", None)
            if ("stream_options" in options
                    and status in (400, 422, 501)
                    and any(word in detail for word in ("stream_options", "include_usage"))):
                del options["stream_options"]
            elif "max_tokens" in options and "max_tokens" in detail:
                # Новые модели OpenAI принимают только max_completion_tokens.
                options["max_completion_tokens"] = options.pop("max_tokens")
            elif "extra_body" in options and any(
                    word in detail for word in ("reasoning_effort", "extra_body")):
                del options["extra_body"]
                if diag is not None:
                    diag["reasoning_rejected"] = True
            elif "frequency_penalty" in options and any(word in detail for word in PENALTY_REJECT):
                del options["frequency_penalty"]
                del options["presence_penalty"]
            else:
                raise


def _start_stream(messages, *, max_tokens=None, reasoning_effort=None, diag=None):
    limit = NARRATION_MAX_TOKENS if max_tokens is None else max_tokens
    effort = CHAT_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    return _create_with_fallback(_narration_options(messages, limit, True, effort), diag)


def _start_completion(messages, *, max_tokens=None, reasoning_effort=None, diag=None):
    """Ответ целиком, без потока: нужен там, где обрыв на середине недопустим."""
    limit = NARRATION_SETUP_TOKENS if max_tokens is None else max_tokens
    effort = CHAT_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    return _create_with_fallback(_narration_options(messages, limit, False, effort), diag)


def _measured_completion(turn_id, stage, **options):
    response = None
    try:
        response = client.chat.completions.create(**options)
        return response
    finally:
        usage = getattr(response, "usage", None)
        extra = options.get("extra_body") or {}
        limit = extra.get("max_completion_tokens") or options.get("max_tokens")
        completion_tokens = getattr(usage, "completion_tokens", None)
        if limit is not None and isinstance(completion_tokens, int) and completion_tokens > limit:
            reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
            app.logger.warning("Completion over limit: app_revision=%s stage=%s model=%s tokens=%d limit=%d reasoning=%s",
                               APP_REVISION, stage, options["model"], completion_tokens, limit, reasoning)
        if turn_id:
            model = getattr(response, "model", None) or options["model"]
            TOKEN_USAGE.record(turn_id, stage, model, usage)


def _header_token(name, limit):
    """Безопасное значение заголовка: только буквы, цифры, дефис и подчёркивание."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", request.headers.get(name, ""))[:limit]


PRESENCE_COOKIE = "rpg_visitor"
PRESENCE_LIMIT = 10
PRESENCE_TIMEOUT = 90
CONVERSATION_WINDOW = 180
_GAME_ENDPOINTS = {
    "api_chat", "api_state", "api_roll", "api_saves_list", "api_save",
    "api_save_rename", "api_save_delete", "api_load",
}


@dataclass
class _Visitor:
    last_seen: float
    last_chat: float | None = None
    active_chats: int = 0


_visitors = {}
_presence_lock = threading.Lock()


def _check_presence(visitor_id, *, admit=False, start_chat=False):
    now = monotonic()
    with _presence_lock:
        expired = [key for key, visitor in _visitors.items()
                   if not visitor.active_chats and now - visitor.last_seen >= PRESENCE_TIMEOUT]
        for key in expired:
            del _visitors[key]

        visitor = _visitors.get(visitor_id)
        if visitor is None and admit and len(_visitors) < PRESENCE_LIMIT:
            visitor = _visitors[visitor_id] = _Visitor(now)
        if visitor is not None:
            visitor.last_seen = now
            if start_chat:
                visitor.active_chats += 1
                visitor.last_chat = now
        return {
            "admitted": visitor is not None,
            "online": len(_visitors),
            "limit": PRESENCE_LIMIT,
            "playing": sum(bool(item.active_chats) or
                           (item.last_chat is not None and now - item.last_chat < CONVERSATION_WINDOW)
                           for item in _visitors.values()),
        }


def _finish_chat_presence(visitor_id):
    with _presence_lock:
        visitor = _visitors.get(visitor_id)
        if visitor is not None and visitor.active_chats:
            visitor.active_chats -= 1
            visitor.last_seen = visitor.last_chat = monotonic()


def _admit_visitor():
    visitor_id = request.cookies.get(PRESENCE_COOKIE, "")
    if not re.fullmatch(r"[0-9a-f]{32}", visitor_id):
        visitor_id = uuid4().hex
    return visitor_id, _check_presence(visitor_id, admit=True)


def _presence_response(body, visitor_id, status=200):
    response = app.make_response((body, status))
    response.headers["Cache-Control"] = "no-store"
    if request.cookies.get(PRESENCE_COOKIE) != visitor_id:
        response.set_cookie(PRESENCE_COOKIE, visitor_id, httponly=True, samesite="Lax",
                            secure=request.is_secure or request.headers.get("X-Forwarded-Proto") == "https")
    return response


@app.before_request
def _require_game_presence():
    if request.endpoint not in _GAME_ENDPOINTS:
        return None
    visitor_id = request.cookies.get(PRESENCE_COOKIE, "")
    if not _check_presence(visitor_id)["admitted"]:
        return jsonify({"error": "Место в игре пока недоступно. Дождись свободного места."}), 503


def _client_id():
    """Идентификатор устройства игрока. Разделяет сохранения между людьми."""
    return _header_token("X-Client-Id", 32) or "shared"


SYSTEM_PROMPT = """
Ты — ИИ-мастер текстовой RPG в жанре тёмного фэнтези, D&D и LitRPG. Веди связное приключение на естественном русском. Управляй миром, NPC и последствиями. Решения, мысли и реплики героя оставляй игроку.

Показывай результат действия и двигай сцену. После провала дай последствие или новый путь вместо повторения той же проверки. Доводи установленные цели до развязки. Не добавляй новую обязательную цель задним числом. Различай подтверждённые факты и догадки. Лист отражает состояние после предыдущего ответа. Используй его числа как старт хода. Уже проведённые удары, лечение, покупки и награды не применяй повторно, даже при уточнении. На вопрос о текущем числе ответь числом из листа, если в этом ходу оно не изменилось. Примеры развития сцены: реакция NPC, зацепка, опасность или новый путь. Одна-две яркие детали сильнее перечня эпитетов. Не пересказывай ход игрока, не нагромождай образы и не повторяй одинаковые детали в соседних сценах. Длину и структуру ответа задаёт блок [ФОРМАТ ОТВЕТА].
Поддерживай канон по листу и памяти. Сохраняй имена, место, условия и награду каждого принятого или предложенного задания. При закрытии сверяй исходные этапы, включая обратную доставку или доклад. Успех части задачи оставляет остальные этапы открытыми. Не смешивай людей и улики из разных мест. Свежие реплики уточняют старое резюме. У важных NPC свои мотивы, тайны и голос. Мир меняется и без героя: держи 1–4 силы с часами 1–6, продвигай их после крупных сцен и показывай последствия через события.

ПРОВЕРКИ И СЛОЖНОСТЬ (DC)
Если исход неясен, а провал что-то меняет, запроси проверку до описания результата. Примеры: прокрасться мимо дозора, обезвредить ловушку, убедить сопротивляющегося NPC. Открыть незапертую дверь или взять видимый предмет можно сразу. Владение навыком и подготовка дают преимущество по ситуации, но не отменяют риск.
Атака по сопротивляющейся цели — против её защиты. Для эффектов со спасброском бросает цель. Правило способности имеет приоритет. Укажи, кто бросает, проверку, DC, общий модификатор и команду, например «Скрытность, DC 13. Команда: /roll 1d20+4». Бонус навыка из листа уже суммарный. Без него используй мод характеристики и мастерство при владении (+2 на ур. 1–4, +3 на 5–8, +4 на 9–12, +5 на 13–16, +6 на 17–20).
DC 5/8/10/12 — низкая сложность, 15 — средняя, 18/20/25/30 — высокая. Навык и характеристика 16+ могут снизить DC на 2–3. Учитывай положение, ранения, темноту, спешку, помощь и укрытие цели (половинное +2, три четверти +5, полное закрывает цель). Если лежит и хочет атаковать, сначала подъём DC 10. Схвачен или связан — освобождение DC 15, если возможно.
Запрашивай один бросок за раз. Пока игрок не прислал [[БРОСОК]], закончи на запросе и не объявляй итог, урон или XP за ожидающее действие. Итог ≥ DC — успех, ниже — провал. Бросок урона или лечения применяй напрямую, без сравнения с DC. Преимущество: /roll 2d20kh1+N, помеха: /roll 2d20kl1+N (подставь суммарный модификатор). Бросок относится к последней проверке: возьми её цель, сложность и тип из непосредственно предыдущего ответа мастера вместе с действием перед ним. Не переноси бросок на другую сцену. Если итог снова приведён в тексте, используй итог калькулятора.

ДЫХАНИЕ ДРАКОНА В БОЮ:
Спасбросок делает каждая цель. DC = 8 + мод Телосложения героя + мастерство. Огонь, кислота и молния — Ловкость, холод и яд — Телосложение. Уточни неизвестную стихию. Игрок бросает за цели по очереди. Назови цель и команду отдельно, например /roll 1d20+2. При неизвестном модификаторе используй +0. Затем запроси один /roll 2d6 на урон: весь при провале, половину с округлением вниз при успехе. Сначала дождись бросков, затем опиши результат и потрать одно применение. Если заряд потрачен, сообщи до бросков. Если целей нет, спасбросок не нужен. Атака героя не бросается.

XP И ЛИСТ
Обычные действия, в том числе открытие двери, сами по себе дают 0 XP, как и отдельный удар, успешный бросок или промежуточный этап. Одно достижение учитывай один раз. За реальную угрозу — по силе: 10–25 / 50–100 / 150–250 / 500–1000 / 1500–2500 / 3000–5000 XP. Победа без убийства даёт половину. За серьёзное препятствие вне боя 50–100, сложное 150–250, эпическое 500+, за решающую зацепку 10–20. За квест выдай обещанную награду на завершении, исключив дубли за тот же результат. Укажи новую награду отдельной строкой в конце: +50 XP. Для задания укажи его точное название из листа: +50 XP — квест «Название». При обсуждении уже полученного опыта не начинай строку со знака плюс. При отсутствии награды пропусти строку. XP и уровень считает сервер. Если в листе есть next_upgrade_level, предложи улучшения для этого уровня: одну характеристику на +1 и новое или улучшенное умение. Дождись явного выбора игрока. После выбора начни отдельную строку с «Повышение до уровня N:» (N — номер уровня), назови характеристику, умение и, если герой получил или улучшил навык, его название и новый бонус. За ход разрешай одно повышение. Пока игрок выбирает, не применяй бонусы.
Письма, записи, карты и слухи — в заметки. Физический экземпляр письма, расписки или накладной, который можно вручить, учитывай также в инвентаре. Золотые, серебряные и медные монеты — разные суммы, сохраняй названную валюту. В рассказе дай суть документа. У квеста есть цель, этапы и награда: main 200–500 XP, side 50–150, personal 80–200, faction 100–250. Завершённый квест закрой; если это цепочка, открой следующий. После описания нового героя создай лист с расой, классом, предысторией, характеристиками и навыками, затем начни сцену.
"""

CHARACTER_CREATION_RULES = """
════════ СОЗДАНИЕ ПЕРСОНАЖА ════════
При первом описании персонажа определи:
1. РАСУ и КЛАСС. Возьми их из свободного описания или указанных полей. Пропущенные выбери сам по смыслу описания и миру.
2. ПРЕДЫСТОРИЮ (Солдат, Преступник, Мудрец, Народный герой, Благородный, Отшельник, Артист, Бродяга, Моряк, Наёмник, Шарлатан, Прислужник).
3. ХАРАКТЕРИСТИКИ (базовые по классу + расовые бонусы).
4. ВЛАДЕНИЯ, НАВЫКИ, УМЕНИЯ.

Выбор игрока, включая авторские варианты, сохраняй дословно: не переименовывай расу, класс и предысторию. Авторские бонусы, названные игроком для своей расы, применяй как есть.

Характеристики считай сам: база по классу плюс расовые бонусы плюс всё, что назвал игрок. Числа из служебного листа не копируй — там заготовка. HP, кошелёк и снаряжение выдай по классу: Колдун получает 2 ячейки заклинаний, лёгкие доспехи, кинжалы и простые инструменты, стартовый кошелёк 10–15 зм.

════════ СТРУКТУРА ПЕРВОГО ОТВЕТА ════════
Оформляй лист и начинай игру в этом ответе. Игрок может указать имя, расу, класс, предысторию и завязку свободным текстом, выбрать отдельные поля или полностью поручить выбор мастеру. Придумай имя, расу, класс или предысторию, если их не указали и они не ясны из текста. Не требуй выбора из вариантов и не задавай уточняющий вопрос перед листом. Если игрок указал стартовую сцену или мир, сохрани их.

Лист героя оформляй одним блоком между метками [[ЛИСТ]] и [[/ЛИСТ]]. Внутри — эти строки, каждая с новой строки, ровно с такими значками и без пропусков:

[[ЛИСТ]]
👤 Кейр — эльф · плут · Преступник
📊 СИЛ 10 (+0) · ЛОВ 17 (+3) · ТЕЛ 12 (+1) · ИНТ 12 (+1) · МДР 10 (+0) · ХАР 10 (+0)
❤️ HP 10/10 · ⭐ Ур. 1 · Опыт 0/100 · 💰 15 зм
🛡 Владения: лёгкие доспехи, рапиры, воровские инструменты
🗣 Языки: общий, эльфийский
🎯 Навыки: Скрытность +5, Обман +2
⚡ Активные: Хитрое действие — 3/короткий отдых
🌀 Пассивные: Скрытая атака 1d6, тёмное зрение
🎒 Инвентарь: рапира, кожаный доспех, воровские инструменты
[[/ЛИСТ]]

Метки [[ЛИСТ]] и [[/ЛИСТ]] стоят отдельными строками. Все девять строк обязательны: если чего-то нет, поставь пустое значение, но строку дай, иначе карточка игрока останется без раздела. Метка [[/ЛИСТ]] обязательна — без неё блок считается оборванным.

Первая строка — имя, раса, класс, предыстория через «·». Характеристики уже с расовыми бонусами, сокращения СИЛ/ЛОВ/ТЕЛ/ИНТ/МДР/ХАР, модификатор — (значение − 10) // 2. В навыках — каждый навык с суммарным бонусом (мод характеристики плюс мастерство +2 на ур. 1–4).

НАЗЫВАЙ КОНКРЕТНЫЕ ЭФФЕКТЫ. В активных умениях пиши, что именно герой может применить и с какой силой, а не общую категорию. Не «Заклинания — 2 ячейки», а «Мистический заряд 1d10 — без ограничений, Порча — 2 ячейки 1 круга / долгий отдых». Не «Второе дыхание — 1/короткий», а «Второе дыхание — 1/короткий отдых, 1d10+1 HP». Не «Ярость — 2/долгий», а «Ярость — 2/долгий отдых, +2 урона, 1 минута». Каждое имя эффекта бери из списка класса и расы ниже и, если у него есть числовой эффект, укажи кубик и длительность. По листу должно быть понятно без вопросов, что можно применить в бою и сколько раз.

То же правило для пассивных: не «Тёмное зрение», а «Тёмное зрение 18 м, скрытая атака 1d6». Если у пассивного эффекта нет числового значения, всё равно назови его конкретно, а не категорией.

В инвентаре — предметы с количеством, там же документы и письма.

Затем, после пустой строки, 1–2 абзаца открывающей сцены по правилам обычного хода, с зацепкой в конце. Раньше блока сцену не начинай.

Тот же блок [[ЛИСТ]] повторяй, если игрок прямо просит показать лист. В обычном рассказе метки [[ЛИСТ]] не используй.

Названия навыков, умений и предметов придумывай один раз: те же слова пойдут в лист игрока. Не выдумывай умений вне класса и расы и не дублируй расовые владения отдельными навыками.

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
"""

# Идёт последним блоком системного сообщения: это правило модель видит
# непосредственно перед ответом, поэтому оно и задаёт длину.
LENGTH_NOTE = """
[ФОРМАТ ОТВЕТА — ГЛАВНОЕ ПРАВИЛО ЭТОГО ХОДА]
Пиши 2 полных абзаца по 4–7 предложений (примерно 900–1400 знаков). Первый: что произошло, как отреагировали NPC и мир, одна-две конкретные детали. Второй: последствия для героя, изменение обстановки и зацепка к следующему шагу. Важная сцена — до 4 абзацев, кульминация — до 6. Ответ в одну-две строки считается ошибкой: если сцена не даёт материала, добавь реакцию мира, деталь окружения и зацепку, но не выдумывай новый обязательный сюжет. Проверку по-прежнему заканчивай запросом броска и без итога. Строку награды (+50 XP — квест «Название») оставляй отдельной последней строкой, а не внутри абзаца. При создании персонажа правило двух абзацев не действует: следуй блоку [СТРУКТУРА ПЕРВОГО ОТВЕТА] — лист в метках [[ЛИСТ]], затем 1–2 абзаца открывающей сцены. Если игрок прямо просит показать лист, тоже дай блок [[ЛИСТ]] и после него обычный ответ.
"""


STATE_EXTRACT_PROMPT = """
Сохраняй канон кампании. Для новой сцены или изменений верни scene: {location,time,participants,goal,situation}. Указывай устойчивое название локации. participants — только те, кто сейчас рядом. При смене локации обнови участников, цель и обстановку сцены. Возвращай только изменения. Для подтверждённых событий, показаний, версий и владения уликами верни обновления facts: [{id,subject,detail,source,holder,status}]. Устойчивый id привязан к одному факту или предмету, сохрани его при уточнении и передаче. status: confirmed, reported, suspected или resolved. Для передачи физической улики обнови holder и inventory_remove в том же ходе. Для новой объявленной команды /roll верни pending_check: {kind,title,target,effect,ability}; kind — check, attack, save, damage, heal или initiative. effect только hero_damage для урона персонажу игрока или hero_heal для его лечения. Для урона NPC effect опусти. Формулу и DC сервер получает из реплики. Если игрок явно отменил прежнее действие и мастер подтвердил отмену, верни cancel_check: true.
Для завершённого отдыха верни rest_event: {kind,evidence}, где kind — short_rest или long_rest, evidence — фраза из новой реплики о завершении отдыха. Для состоявшегося использования умения без отдельного броска верни ability_event: {name,evidence}, где name совпадает с карточкой и evidence описывает применение в этом ходе. Дыхание дракона и Второе дыхание с проверкой расходует сервер. При изменении описания существующего умения сохраняй текущие заряды.
При покупке верни purchase: {item,quantity,currency,price,evidence}, где item — точное название, quantity — число полученных предметов, currency — gold, silver или copper, price — общая цена, evidence — цитата новой подтверждённой покупки. Для покупки используй только purchase, не возвращай отдельные изменения кошелька или инвентаря.

Обнови лист RPG по последнему ходу. Верни строго JSON только изменённых полей, {} при отсутствии изменений.
Меняй состояние только по новому результату в последней реплике мастера. Предыдущие реплики и пересказ старых событий служат контекстом. Лист уже учитывает прошлые удары, лечение, покупки и расходы. Изменяй кошелёк и инвентарь лишь при реальном новом обмене, расходе или находке. Пересказ, проверка числа, предложения и суммы до прошлого обмена ничего не меняют. Для существующего героя верни hp_delta (отрицательный урон или положительное лечение) только за новый результат, и hp_event: короткая дословная фраза из последнего ответа, где этот урон или лечение произошли, а не где обсуждается итоговое здоровье. На запросе броска, сверке или пересказе не возвращай hp_delta. При создании используй hp и max_hp. Для изменений инвентаря или монет у существующего героя верни resource_events: {"inventory":"дословная фраза нового получения, траты или передачи", "wallet":"дословная фраза новой оплаты"}. Укажи только затронутые группы. Цифры в пересказе или сверке не являются событиями. xp, xp_next, level, level_pending, xp_event_ids считает сервер.
Имя, раса, класс и предыстория меняются только по событию. Поля: name, race, class, background, hp, max_hp, gold, silver, copper, stats {str,dex,con,int,wis,cha}. HP не выше max_hp. Базовые характеристики, владения, навыки и умения назначай при создании, беря их из блока [[ЛИСТ]]: строка «📊» — характеристики, «❤️» — hp, уровень, опыт и кошелёк, «🛡» — владения, «🗣» — языки, «🎯» — навыки с суммарным бонусом, «⚡» — активные умения с uses_max, uses_left и recharge (short_rest, long_rest или day), «🌀» — пассивные и постоянные эффекты с recharge passive и без uses_max, «🎒» — инвентарь. Значки и метки [[ЛИСТ]] в лист не переноси. Название умения бери из блока как есть, включая числовой эффект. Инвентарь, кошелёк и характеристики из блока переноси сразу, в том же ходе. Навык {name,bonus,source}; умение {name,description,uses_max,uses_left,recharge}. Трать применения по событию и восстанавливай по отдыху. Для Дыхания дракона спиши одно применение после разрешения урона, а не пока мастер ждёт бросок. Сохраняй полное описание эффекта при изменении только зарядов. Активное умение «без ограничений» передавай с recharge: "unlimited", без uses_max и uses_left. Пассивный эффект передавай с recharge: "passive". Не используй отрицательные заряды, Markdown и HTML-сущности в названии.
Изменённую запись коллекции возвращай целиком с прежним id или именем. inventory содержит только добавленные или изменённые вещи. Для удаления верни inventory_remove: [{"name":"точное имя из листа","quantity":1}]. Удаляй за фактический расход, передачу или потерю. Для замены старого названия новым явно удали старую запись. Количество у изменённого предмета указывай итоговое. notes — содержание документов, писем, слухов и записей. Полученный физический документ также добавь в inventory, при вручении удали его из inventory. Заметка {id,type,title,content,source,date,unread}; type: note, letter, document, notification, rumor или journal.
Квест {id,name,type,status,chain,chain_order,stages,current_stage,rewards,giver,location}; stages — объекты {text,done}; rewards — объект {xp,gold,silver,copper,items,reputation}. Записывай окончательно предложенные условия со статусом offered, принятые — active. Сохраняй id, название, этапы и награду. Отмечай каждый фактически выполненный этап по отдельности. Закрывай квест после сдачи всех обязательных этапов. Если квест официально закрыли с неполным результатом, верни failed и сохрани невыполненные пункты. Исправляй явно отменённое закрытие с прежним id.
Временные поля по сцене: conditions, position, cover, weapon_in_hand, restrained, dc_modifier, dc_modifier_reasons. Отрази уже совершённое перемещение даже перед запросом броска. При смене сцены обнови cover: "нет", если герой покинул прежнее укрытие без нового. При пустых руках верни weapon_in_hand: "". Если условия прошли, верни пустой список; при отсутствии помех — 0 и []. Связи {name,attitude,note}; faction_clocks {name,progress,max,goal}. При обучении или событии в мире обновляй skills и abilities даже после создания героя.
Повышение передавай отдельным полем level_upgrade только когда игрок явно выбрал улучшения и мастер подтвердил их в последнем ответе строкой «Повышение до уровня N:». Формат: {"level_upgrade":{"level":2,"stat":"str","ability":{"name":"Рывок","description":"...","uses_max":1,"uses_left":1,"recharge":"short_rest"},"skills":[{"name":"Атлетика","bonus":4,"source":"Воин"}]}}. Используй next_upgrade_level из листа. Для улучшения существующего умения передай его точное имя и новый максимум, сервер сохранит число потраченных зарядов. skills включай, если выбран или улучшен навык. Не дублируй бонусы повышения в других полях JSON. level_history записывает сервер.
"""


SUMMARY_PROMPT = """
Обнови краткую память кампании по прежнему резюме и новым событиям. Сохрани имена и принадлежность NPC, решения, обещания, условия и награды открытых квестов, важные предметы и зацепки. Привязывай события к месту и квесту, отмечай уже закрытые задачи и выданные награды. Укажи последнее место и открытую угрозу. Замени устаревшее, объедини дубли. Не сочиняй нового. До 10 коротких пунктов, без пересказа реплик, описаний внешности, бросков и текущих чисел листа.
"""

DICE_RE = re.compile(r"^\s*(\d*)d(\d+)(?:(kh|kl)(\d+))?([+-]\d+)?\s*$", re.IGNORECASE)

DEFAULT_STATE = {
    "name": "Пустой", "race": "", "class": "", "background": "",
    "level": 1, "xp": 0, "xp_next": 100,
    "hp": 10, "max_hp": 10, "gold": 0, "silver": 0, "copper": 0,
    "stats": {"str": 10, "dex": 10, "con": 10, "int": 10, "wis": 10, "cha": 10},
    "proficiencies": [], "skills": [], "abilities": [],
    "inventory": [], "notes": [], "quests": [], "conditions": [],
    "relationships": [], "faction_clocks": [],
    "dc_modifier": 0, "dc_modifier_reasons": [],
    "position": "стоит", "cover": "нет",
    "weapon_in_hand": "", "restrained": False,
    "scene": {}, "facts": [],
    "level_history": [], "xp_event_ids": [], "quest_xp_awards": [], "level_pending": 0
}

# Основные поля листа и краткие условия текущих квестов для мастера.
STATE_FIELDS = ("name", "race", "class", "background", "level", "xp", "xp_next",
                "hp", "max_hp", "gold", "silver", "copper", "stats", "conditions", "position",
                "cover", "weapon_in_hand", "restrained")
STATE_NOTE = ("[ЛИСТ ПОСЛЕ ПРЕДЫДУЩЕГО ОТВЕТА МАСТЕРА. "
              "Текущий ход игрока ещё не учтён. Используй числа отсюда.]\n")

# На первом ходу лист ещё пуст: числа в state_context — заготовка DEFAULT_STATE,
# и подсказка «используй числа отсюда» заставляла модель скопировать десятки.
STATE_NOTE_SETUP = ("[ПЕРСОНАЖ ЕЩЁ НЕ СОЗДАН. Числа заготовки сюда не подставлены. "
                    "Характеристики с расовыми бонусами, HP, кошелёк и снаряжение "
                    "назначь сам по расе, классу и предыстории.]\n")


KEEP_RECENT = 6
SUMMARY_BATCH_MESSAGES = 4
SUMMARY_MAX_BATCH_MESSAGES = 32
RECENT_HISTORY_CHARS = max(800, int(os.getenv("RECENT_HISTORY_CHARS", "2400")))
SUMMARY_INPUT_CHARS = max(600, int(os.getenv("SUMMARY_INPUT_CHARS", "1800")))
STATE_HISTORY_CHARS = 1800
SUMMARY_MAX_TOKENS = max(128, int(os.getenv("SUMMARY_MAX_TOKENS", "400")))
SUMMARY_MAX_SESSIONS = int(os.getenv("SUMMARY_MAX_SESSIONS", "128"))

# Резюме прошлого живёт на сервере, по session_id. Клиент его не хранит
# и не присылает — иначе оно пересчитывалось бы каждый ход.
_SUMMARY_STATE = {}
_SUMMARY_LOCK = threading.Lock()
_SUMMARY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rpg-summary")
_SUMMARY_JOB = threading.BoundedSemaphore(1)


def _summary_get(sid):
    with _SUMMARY_LOCK:
        st = _SUMMARY_STATE.get(sid)
        return dict(st) if st else {"text": "", "folded": 0}


def _summary_put(sid, text, folded):
    with _SUMMARY_LOCK:
        if sid not in _SUMMARY_STATE and len(_SUMMARY_STATE) >= SUMMARY_MAX_SESSIONS:
            _SUMMARY_STATE.pop(next(iter(_SUMMARY_STATE)))
        _SUMMARY_STATE[sid] = {"text": text, "folded": folded}


def _budgeted_dialogue(messages, max_chars, preserve_latest=False):
    """Сохраняет формат и порядок реплик, сокращая самые длинные фрагменты."""
    if not messages:
        return []
    recent = messages[-1:] if preserve_latest else []
    earlier = messages[:-1] if recent else messages
    available = max(0, max_chars - sum(len(m.get("content", "")) for m in recent))
    if available < len(earlier) * 40:
        keep = available // 40
        earlier = earlier[-keep:] if keep else []
    ordered = sorted(range(len(earlier)), key=lambda index: len(earlier[index].get("content", "")))
    limits = [0] * len(earlier)
    for position, index in enumerate(ordered):
        limit = min(len(earlier[index].get("content", "")), available // (len(ordered) - position))
        limits[index] = limit
        available -= limit
    clipped = []
    for message, limit in zip(earlier, limits):
        content = message.get("content", "")
        if len(content) > limit:
            excerpt_chars = limit - len(" […] ")
            head = excerpt_chars * 2 // 3
            tail = excerpt_chars - head
            content = content[:head] + " […] " + content[-tail:]
        clipped.append({**message, "content": content})
    return clipped + recent


def _dialogue_messages(messages):
    return [{"role": message["role"], "content": message["content"]} for message in messages
            if isinstance(message, dict) and message.get("role") in ("user", "assistant")
            and isinstance(message.get("content"), str)]


def _recent_dialogue(dialogue):
    """Приоритет у непрерывного хвоста: действие, последняя проверка и ответ."""
    remaining = RECENT_HISTORY_CHARS
    tail = []
    for message in reversed(dialogue):
        content = message.get("content", "")
        if len(content) > remaining:
            if len(tail) < 3 and remaining >= 160:
                head = remaining // 2
                end = remaining - head - len(" […] ")
                tail.append({**message, "content": content[:head] + " […] " + content[-end:]})
            break
        tail.append(message)
        remaining -= len(content)
    return list(reversed(tail))


def _messages_with_memory(messages, session_id, *, character_setup=False):
    summary = _summary_get(session_id)
    dialogue = _dialogue_messages(messages)
    parts = [message["content"] for message in messages
             if message.get("role") == "system"
             and not message["content"].startswith(
                 ("[РЕЗЮМЕ ПРОШЛОГО]", STATE_NOTE, STATE_NOTE_SETUP))]
    if summary["text"]:
        parts.append(f"[РЕЗЮМЕ ПРОШЛОГО]\n{summary['text']}")
    parts.extend(message["content"] for message in messages
                 if message.get("role") == "system"
                 and message["content"].startswith((STATE_NOTE, STATE_NOTE_SETUP)))
    system = [{"role": "system", "content": "\n\n".join(parts)}] if parts else []
    if character_setup:
        return system + dialogue[-1:]
    # Даже для старого сохранения без резюме ближайшая проверка остаётся целиком.
    tail = _recent_dialogue(dialogue[max(0, min(summary["folded"], len(dialogue) - 3)):])
    return system + tail


def _refresh_summary(dialogue, session_id, turn_id):
    summary = _summary_get(session_id)
    folded = summary["folded"]
    target = min(len(dialogue) - KEEP_RECENT, folded + SUMMARY_MAX_BATCH_MESSAGES)
    if target - folded < SUMMARY_BATCH_MESSAGES:
        return
    recent = _budgeted_dialogue(dialogue[folded:target], SUMMARY_INPUT_CHARS)
    new_text = "\n".join(f"{message['role']}: {message['content']}" for message in recent if message.get("content"))
    if not new_text:
        return

    previous = summary["text"]
    archive = f"{previous}\n\nНОВЫЕ СОБЫТИЯ:\n{new_text}" if previous else new_text
    try:
        with llm_slot(timeout=0) as ok:
            if not ok:
                return
            limit_options = (_qwen_service_options(SUMMARY_MODEL, QWEN_SUMMARY_LIMIT)
                             or {"max_tokens": SUMMARY_MAX_TOKENS})
            resp = _measured_completion(
                turn_id, "summary",
                model=SUMMARY_MODEL,
                messages=[{"role": "system", "content": SUMMARY_PROMPT},
                          {"role": "user", "content": archive}],
                temperature=0.2,
                timeout=min(LLM_TIMEOUT, 20),
                **limit_options,
            )
        choice = resp.choices[0]
        summary_text = (choice.message.content or "").strip()
        if summary_text and getattr(choice, "finish_reason", None) != "length":
            _summary_put(session_id, summary_text, target)
    except Exception as exc:
        app.logger.warning("Не удалось обновить резюме: %s", type(exc).__name__)


def _run_summary_job(dialogue, session_id, turn_id):
    try:
        with app.app_context():
            _refresh_summary(dialogue, session_id, turn_id)
    finally:
        _SUMMARY_JOB.release()


def _schedule_summary(messages, session_id, turn_id):
    if not session_id:
        return
    dialogue = _dialogue_messages(messages)
    if len(dialogue) - _summary_get(session_id)["folded"] < KEEP_RECENT + SUMMARY_BATCH_MESSAGES:
        return
    # No backlog: busy chat traffic can delay a refresh until a following turn.
    if not _SUMMARY_JOB.acquire(blocking=False):
        return
    try:
        _SUMMARY_EXECUTOR.submit(_run_summary_job, dialogue, session_id, turn_id)
    except RuntimeError:
        _SUMMARY_JOB.release()



def roll_dice(expr):
    m = DICE_RE.match(expr or "")
    if not m:
        raise ValueError("Формат: 1d20+3, 2d6, d100-1, 2d20kh1+1 или 2d20kl1+1")
    count = int(m.group(1) or "1")
    sides = int(m.group(2))
    keep_mode, keep_count_text, modifier = m.groups()[2:]
    mod = int(modifier or "0")
    if not (1 <= count <= 100):
        raise ValueError("Кубиков 1..100")
    if not (2 <= sides <= 1000):
        raise ValueError("Сторон 2..1000")
    keep_count = int(keep_count_text) if keep_mode else count
    if not 1 <= keep_count <= count:
        raise ValueError("Число учитываемых кубиков должно быть от 1 до числа бросков")
    rolls = [random.randint(1, sides) for _ in range(count)]
    kept = sorted(rolls, reverse=keep_mode.lower() == "kh")[:keep_count] if keep_mode else rolls
    result = {"expr": expr, "rolls": rolls, "mod": mod, "total": sum(kept) + mod}
    if keep_mode:
        result["kept"] = kept
    return result


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


def _quest_name(quest):
    name = quest.get("name") if isinstance(quest, dict) else quest
    return " ".join(name.casefold().split()) if isinstance(name, str) else ""


def _quest_steps_open(quest):
    stages = quest.get("stages")
    return isinstance(stages, list) and any(isinstance(stage, dict) and not stage.get("done") for stage in stages)


def _merge_quests(old, changes):
    merged = [quest for value in old if (quest := normalize_quest(value))]
    for change in changes:
        if not isinstance(change, dict) or not _quest_name(change):
            continue
        name = _quest_name(change)
        index = next((i for i, quest in enumerate(merged)
                      if (change.get("id") and quest.get("id") == change["id"])
                      or _quest_name(quest) == name), None)
        if index is None:
            item = {**change, "id": change.get("id") or uuid4().hex}
            if item.get("status") == "completed" and _quest_steps_open(item):
                item["status"] = "active"
            merged.insert(0, item)
            continue
        previous = merged.pop(index)
        updated = {**previous, **change, "id": previous.get("id") or change.get("id")}
        if previous.get("status") in {"completed", "failed"}:
            updated["status"] = previous["status"]
        if previous.get("rewards"):
            updated["rewards"] = previous["rewards"]
        previous_stages = previous.get("stages")
        changed_stages = change.get("stages")
        if isinstance(previous_stages, list) and previous_stages:
            def stage_key(stage):
                return _quest_name(stage.get("text")) if isinstance(stage, dict) else ""

            patches = {stage_key(stage): stage for stage in (changed_stages or [])
                       if stage_key(stage)} if isinstance(changed_stages, list) else {}
            stages = []
            for stage in previous_stages:
                key = stage_key(stage)
                patch = patches.pop(key, None)
                if key:
                    stage = dict(stage)
                    stage["id"] = stage.get("id") or uuid4().hex
                    if patch and patch.get("done") is True:
                        stage["done"] = True
                stages.append(stage)
            updated["stages"] = stages
        if (previous.get("status") in {"offered", "active"} and updated.get("status") == "completed"
                and _quest_steps_open(updated)):
            updated["status"] = "active"
        merged.insert(0, updated)
    # Older versions could assign multiple IDs to the same quest.
    seen = set()
    result = []
    for quest in merged:
        name = _quest_name(quest)
        if name and name in seen:
            continue
        seen.add(name)
        result.append(quest)
    return result


def _quest_context(quests, awarded):
    if not isinstance(quests, list):
        return []
    awarded = set(awarded) if isinstance(awarded, list) and all(isinstance(name, str) for name in awarded) else set()
    result = []
    for quest in quests:
        if not isinstance(quest, dict) or not isinstance(quest.get("name"), str):
            continue
        status = quest.get("status", "active")
        if status not in {"offered", "active", "completed", "failed"}:
            continue
        stages = quest.get("stages")
        entry = {"name": quest["name"][:100], "status": status}
        rewards = quest.get("rewards")
        if isinstance(rewards, dict):
            entry["rewards"] = {key: value for key, value in rewards.items()
                                if key in {"xp", "gold", "silver", "copper"} and type(value) is int}
        elif isinstance(rewards, list):
            entry["rewards"] = [item[:45] for item in rewards[:3] if isinstance(item, str)]
        if _quest_name(quest) in awarded:
            entry["xp_paid"] = True
        if status in {"offered", "active"} and isinstance(stages, list):
            entry["open"] = [stage["text"][:130] for stage in stages
                             if isinstance(stage, dict) and not stage.get("done")
                             and isinstance(stage.get("text"), str)][:4]
        result.append(entry)
    result.sort(key=lambda q: q["status"] in {"completed", "failed"})
    excerpt = []
    budget = 760
    for quest in result[:6]:
        size = len(json.dumps(quest, ensure_ascii=False))
        if size <= budget:
            excerpt.append(quest)
            budget -= size
    return excerpt


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


XP_EVENT_RE = re.compile(
    r"^[ \t]*(?:\*\*)?\+[ \t]*(\d{1,7})[ \t]*XP(?:\*\*)?"
    r"(?:[ \t]+[—–-][ \t]+[^\r\n]+|[.])?[ \t]*$", re.IGNORECASE | re.MULTILINE
)
INLINE_XP_EVENT_RE = re.compile(
    r"[.!?][ \t]+(?:\*\*)?\+[ \t]*(\d{1,7})[ \t]*XP(?:\*\*)?\.?\s*\Z",
    re.IGNORECASE,
)
QUEST_AWARD_RE = re.compile(r"[—–-][ \t]+квест[ \t]+«([^»\r\n]+)»[.!]?[ \t]*$", re.IGNORECASE)
LEVEL_UPGRADE_RE = re.compile(r"^(?:\*\*)?Повышение до уровня[ \t]+(\d{1,2})(?:\*\*)?:", re.IGNORECASE | re.MULTILINE)


def message_event_id(message, index):
    raw = f"{index}\0{message.get('role','')}\0{message.get('content','')}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _awards_in_reply(text, allow_inline=False):
    lines = [line.strip().replace("**", "") for line in text.splitlines()]
    last = lines[-1] if lines else ""
    quest = QUEST_AWARD_RE.search(last)
    if quest and not last.startswith("+"):
        inline = re.search(r"[.!?]\s+\+\s*(\d{1,7})\s*XP\s*$", last[:quest.start()], re.I)
        if inline and allow_inline:
            return [(int(inline[1]), _quest_name(quest[1]))]
    matches = [(int(match[1]), _quest_name(quest[1]) if (quest := QUEST_AWARD_RE.search(match[0])) else "")
               for match in XP_EVENT_RE.finditer(text)]
    inline = INLINE_XP_EVENT_RE.search(text) if allow_inline else None
    if inline:
        matches.append((int(inline[1]), ""))
    if matches and len(lines) >= 2 and not matches[-1][1]:
        heading = re.fullmatch(r"Квест «([^»]+)» заверш[её]н[.!]?", lines[-2], re.I)
        if heading:
            matches[-1] = (matches[-1][0], _quest_name(heading[1]))
    return matches


def _item_name(item):
    if isinstance(item, dict):
        item = item.get("name")
    return " ".join(item.casefold().split()) if isinstance(item, str) else ""


def _remove_inventory_items(items, removals):
    if not isinstance(removals, list):
        return items
    remaining = list(items)
    for removal in removals:
        if not isinstance(removal, dict):
            continue
        name = _item_name(removal)
        quantity = removal.get("quantity", 1)
        if not name or type(quantity) is not int or quantity <= 0:
            continue
        updated = []
        for item in remaining:
            if quantity <= 0 or _item_name(item) != name:
                updated.append(item)
                continue
            count = item.get("quantity", 1) if isinstance(item, dict) else 1
            count = count if type(count) is int and count > 0 else 1
            removed = min(quantity, count)
            quantity -= removed
            if count > removed:
                updated.append({**item, "quantity": count - removed})
        remaining = updated
    return remaining


def _next_upgrade_level(state):
    pending = int(state.get("level_pending", 0) or 0)
    level = max(1, int(state.get("level", 1) or 1))
    if pending <= 0 or level <= 1:
        return None
    history = state.get("level_history")
    history = history if isinstance(history, list) else []
    applied = {record["level"] for record in history
               if isinstance(record, dict) and isinstance(record.get("level"), int)}
    return next((number for number in range(max(2, level - pending + 1), level + 1)
                 if number not in applied), None)


def _update_named_entry(entries, change):
    name = change["name"].strip()
    for index, entry in enumerate(entries):
        if isinstance(entry, dict) and str(entry.get("name", "")).strip().casefold() == name.casefold():
            entries[index] = {**entry, **change, "name": name}
            return
    entries.append({**change, "name": name})


_SKILL_STATS = {
    "атлетика": "str", "акробатика": "dex", "ловкость рук": "dex", "скрытность": "dex",
    "магия": "int", "история": "int", "расследование": "int", "анализ": "int",
    "природа": "int", "религия": "int", "восприятие": "wis", "внимательность": "wis",
    "проницательность": "wis", "медицина": "wis", "выживание": "wis",
    "животные": "wis", "уход за животными": "wis", "обман": "cha",
    "запугивание": "cha", "убеждение": "cha", "выступление": "cha",
}


def _apply_level_upgrade(state, upgrade, next_level):
    if next_level is None or not isinstance(upgrade, dict) or upgrade.get("level") != next_level:
        return False

    stat = upgrade.get("stat")
    ability = normalize_ability(upgrade.get("ability"))
    if not isinstance(stat, str) or stat not in DEFAULT_STATE["stats"] or not isinstance(ability, dict):
        return False
    if not isinstance(ability.get("name"), str) or not ability["name"].strip():
        return False
    if not isinstance(ability.get("description"), str) or not ability["description"].strip():
        return False

    history = state.get("level_history")
    history = list(history) if isinstance(history, list) else []
    if any(isinstance(entry, dict) and entry.get("level") == next_level for entry in history):
        return False

    abilities = state.get("abilities")
    abilities = list(abilities) if isinstance(abilities, list) else []
    name = ability["name"].strip().casefold()
    previous = next((entry for entry in abilities
                     if isinstance(entry, dict) and str(entry.get("name", "")).strip().casefold() == name), {})
    old_limit, old_remaining, new_limit = previous.get("uses_max"), previous.get("uses_left"), ability.get("uses_max")
    if all(type(value) is int and value >= 0 for value in (old_limit, old_remaining, new_limit)):
        spent = max(0, old_limit - old_remaining)
        ability = {**ability, "uses_left": max(0, new_limit - spent)}
    _update_named_entry(abilities, ability)
    state["abilities"] = abilities

    existing_stats = state.get("stats")
    stats = dict(existing_stats) if isinstance(existing_stats, dict) else dict(DEFAULT_STATE["stats"])
    previous_stat = int(stats.get(stat, 10))
    stats[stat] = previous_stat + 1
    state["stats"] = stats
    modifier_gain = (stats[stat] - 10) // 2 - (previous_stat - 10) // 2

    skills = state.get("skills")
    skills = list(skills) if isinstance(skills, list) else []
    if modifier_gain:
        for index, skill in enumerate(skills):
            if not isinstance(skill, dict) or type(skill.get("bonus")) is not int:
                continue
            name = str(skill.get("name", "")).strip().casefold()
            if skill.get("stat", _SKILL_STATS.get(name)) == stat:
                skills[index] = {**skill, "bonus": skill["bonus"] + modifier_gain}

    skill_changes = upgrade.get("skills")
    if isinstance(skill_changes, list):
        for skill in skill_changes:
            if (isinstance(skill, dict) and isinstance(skill.get("name"), str)
                    and skill["name"].strip() and type(skill.get("bonus")) is int):
                _update_named_entry(skills, skill)
    state["skills"] = skills

    hp_gain = max(1, 2 + (int(stats.get("con", 10)) - 10) // 2)
    state["max_hp"] = max(1, int(state.get("max_hp", 1) or 1)) + hp_gain
    state["hp"] = state["max_hp"]
    history.append({"level": next_level, "hp": f"+{hp_gain}", "stats": f"+1 {stat}",
                    "abilities": ability["name"].strip()})
    state["level_history"] = history
    return True


def _grounded_event(evidence, latest):
    if not isinstance(evidence, str):
        return False
    phrase = evidence.strip()
    return len(phrase) >= 5 and phrase in latest


class StateUnavailable(Exception):
    pass


def _signed_roll(message, state, history):
    if not message or not isinstance(message.get("roll_receipt"), str):
        return None
    try:
        roll = ROLL_SIGNER.loads(message["roll_receipt"])
    except BadSignature as exc:
        raise ValueError("Бросок не удалось подтвердить. Обновите ход и повторите бросок.") from exc
    return roll_result(active_check(state, history), roll)


def _purchase(current, change, latest):
    if not isinstance(change, dict) or not _grounded_event(change.get("evidence"), latest):
        return False
    currency, price, quantity = (change.get(key) for key in ("currency", "price", "quantity"))
    name = change.get("item")
    if (currency not in ("gold", "silver", "copper") or type(price) is not int or price <= 0
            or type(quantity) is not int or quantity <= 0 or not _item_name(name)):
        raise StateUnavailable("Не удалось сверить цену или количество покупки.")
    price_before = current.get(currency, 0)
    if type(price_before) is not int or price_before < price:
        raise StateUnavailable("В листе недостаточно монет для описанной покупки.")
    items = list(current.get("inventory") or [])
    index = next((i for i, item in enumerate(items) if _item_name(item) == _item_name(name)), None)
    if index is None:
        items.append({"name": name.strip(), "quantity": quantity})
    else:
        item = items[index]
        item = dict(item) if isinstance(item, dict) else {"name": item}
        item["quantity"] = item.get("quantity", 1) + quantity
        items[index] = item
    current["inventory"], current[currency] = items, price_before - price
    return True


def extract_state(messages, current_state, turn_id=None, result=None):
    current = dict(DEFAULT_STATE)
    if isinstance(current_state, dict):
        current.update(current_state)
    current["abilities"] = normalize_abilities(current.get("abilities"))
    next_upgrade_level = _next_upgrade_level(current)
    last_index = len(messages) - 1
    last_reply = messages[-1] if messages and messages[-1].get("role") == "assistant" else None
    last_event_id = message_event_id(last_reply, last_index) if last_reply else None
    if last_event_id and current.get("last_state_event_id") == last_event_id:
        return current
    current.setdefault("xp_event_ids", [])
    if not isinstance(current.get("xp_event_ids"), list):
        current["xp_event_ids"] = []

    # On legacy saves, mark historical XP-bearing lines as already seen, except
    # the latest assistant message (which may be the just-finished turn).
    if "xp_event_ids" not in (current_state or {}):
        for index, m in enumerate(messages):
            if m.get("role") == "assistant" and XP_EVENT_RE.search(m.get("content", "")) and index != (len(messages) - 1):
                current["xp_event_ids"].append(message_event_id(m, index))

    known_events = {event for event in current["xp_event_ids"] if isinstance(event, str)}
    awarded_messages = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        awards = _awards_in_reply(message.get("content", ""), allow_inline=index == last_index)
        if awards:
            awarded_messages.append((message_event_id(message, index), awards))

    if len(known_events) >= 500:
        first_known = next((i for i, (event_id, _) in enumerate(awarded_messages)
                            if event_id in known_events), 0)
        known_events.update(event_id for event_id, _ in awarded_messages[:first_known])

    fresh_events = [(event_id, amount) for event_id, amount in awarded_messages
                    if event_id not in known_events]

    recent = _budgeted_dialogue(messages[-3:], STATE_HISTORY_CHARS, preserve_latest=True)
    latest = recent[-1].get("content", "") if recent and recent[-1].get("role") == "assistant" else ""
    context_messages = recent[:-1] if latest else recent
    action = ""
    if context_messages and context_messages[-1].get("role") == "user":
        action = context_messages[-1].get("content", "")
        context_messages = context_messages[:-1]
    context = "\n".join(f"{m['role']}: {m['content']}" for m in context_messages if m.get("content"))
    model_state = {key: value for key, value in current.items()
                   if key not in {"xp_event_ids", "quest_xp_awards", "quest_xp_paid_ids",
                                  "last_state_event_id", "level_history"}}
    model_state["facts"] = facts_context(current.get("facts"), action + " " + latest)
    if result:
        model_state["roll_result"] = result
        model_state["rule_effect"] = rules_result(current, result)
    if next_upgrade_level is not None:
        model_state["next_upgrade_level"] = next_upgrade_level
    if isinstance(current.get("quests"), list):
        model_state["quests"] = [
            {key: quest[key] for key in ("id", "name", "status") if key in quest}
            if isinstance(quest, dict) and quest.get("status") in {"completed", "failed"} else quest
            for quest in current["quests"]
        ]
    if isinstance(current.get("notes"), list):
        # Полные тексты остаются в листе. Имена и ID помогают избежать дублей.
        model_state["notes"] = [
            {key: note[key] for key in ("id", "type", "title") if key in note}
            if isinstance(note, dict) else note
            for note in current["notes"]
        ]
    model_inventory = model_state.get("inventory")
    if isinstance(model_inventory, list):
        # Для дельты достаточно имени и количества: описания предметов не нужны.
        model_state["inventory"] = [
            {key: item[key] for key in ("name", "quantity") if key in item}
            if isinstance(item, dict) else item
            for item in model_inventory
        ][:40]
    prompt = (f"ТЕКУЩЕЕ СОСТОЯНИЕ:\n{json.dumps(model_state, ensure_ascii=False)}\n\n"
              f"ПРЕДЫДУЩИЙ КОНТЕКСТ (события уже учтены):\n{context}\n\n"
              f"ТЕКУЩЕЕ ДЕЙСТВИЕ ИГРОКА:\n{action}\n\n"
              f"НОВАЯ РЕПЛИКА МАСТЕРА:\n{latest}\n\nВерни JSON только с изменёнными полями.")
    new_character = (current.get("character_setup_pending")
                     or not (current.get("race") and current.get("class")))
    state_limit = QWEN_STATE_SETUP_LIMIT if new_character or next_upgrade_level is not None else QWEN_STATE_LIMIT
    state_options = (_qwen_service_options(EXTRACT_MODEL, state_limit)
                     or {"max_tokens": state_limit})
    try:
        with llm_slot() as ok:
            if not ok:
                raise RuntimeError("нет свободного слота модели")
            resp = _measured_completion(turn_id, "state",
                model=EXTRACT_MODEL,
                messages=[{"role": "system", "content": STATE_EXTRACT_PROMPT},
                         {"role": "user", "content": prompt}],
                temperature=0.1,
                **state_options,
            )
        state_choice = resp.choices[0]
        if getattr(state_choice, "finish_reason", None) == "length":
            raise ValueError("обновление листа обрезано по лимиту")
        new_state = clean_json(state_choice.message.content or "")
    except Exception as exc:
        app.logger.warning("Извлечение состояния не удалось: %s", _short_error(exc))
        raise StateUnavailable("Лист не обновился. Можно повторить сохранение этого хода.") from exc

    if new_character:
        identity = initial_identity(latest, current)
        if not identity:
            raise StateUnavailable("В ответе мастера не удалось прочитать имя, расу, класс и предысторию.")
        new_state.update(identity)
    campaign_scene = merge_scene(current.get("scene"), new_state.pop("scene", None))
    campaign_facts = merge_facts(current.get("facts"), new_state.pop("facts", None))
    proposed = new_state.pop("pending_check", None)
    cancel_check = new_state.pop("cancel_check", None) is True
    action_index = last_index - 1
    history = messages[:max(0, action_index)]
    previous_check = active_check(current, history)
    next_check = pending_check(latest, proposed, campaign_scene)
    if next_check and last_event_id:
        next_check["id"] = last_event_id[:32]
    if not cancel_check and previous_check and not result:
        # An unrelated answer may restate the same check. Keep its ID and formula.
        if not next_check or next_check["scene_id"] == previous_check.get("scene_id"):
            next_check = previous_check
    previous_scene = current.get("scene")
    if (isinstance(previous_scene, dict) and previous_scene.get("location")
            and campaign_scene.get("location") != previous_scene["location"]):
        new_state.setdefault("position", "стоит")
        new_state.setdefault("cover", "нет")
    new_state["scene"] = campaign_scene
    new_state["facts"] = campaign_facts
    rest_event = new_state.pop("rest_event", None)
    ability_event = new_state.pop("ability_event", None)
    rest = (rest_event.get("kind") if isinstance(rest_event, dict)
            and _grounded_event(rest_event.get("evidence"), latest) else "")
    used_ability = (ability_event.get("name") if isinstance(ability_event, dict)
                    and _grounded_event(ability_event.get("evidence"), latest) else "")
    resolved_check = result.get("check", {}) if result else {}
    resolved_check = resolved_check or {}
    if resolved_check.get("kind") == "damage" and resolved_check.get("ability") == "Дыхание дракона":
        used_ability = "Дыхание дракона"
    if (next_check and next_check != previous_check and next_check.get("kind") == "heal"
            and next_check.get("ability") == "Второе дыхание"):
        used_ability = "Второе дыхание"
        next_check["charge_spent"] = True
        if "воин" in str(current.get("class", "")).casefold() and not new_character:
            next_check["expr"] = f"1d10+{current.get('level', 1)}"
    elif resolved_check.get("kind") == "heal" and resolved_check.get("ability") == "Второе дыхание":
        used_ability = "" if resolved_check.get("charge_spent") else "Второе дыхание"
    if "abilities" in new_state:
        new_state["abilities"] = normalize_abilities(new_state["abilities"])
    if not new_character:
        rest = rest if rest in ("short_rest", "long_rest") else ""
        used_ability = used_ability if isinstance(used_ability, str) else ""
        before_abilities = current.get("abilities") or []
        changed_abilities = new_state.get("abilities")
        if isinstance(changed_abilities, list):
            existing = {_item_name(entry): entry for entry in before_abilities if isinstance(entry, dict)}
            for entry in changed_abilities:
                if not isinstance(entry, dict):
                    continue
                previous = existing.get(_item_name(entry))
                if previous:
                    for field in ("uses_left", "uses_max", "recharge"):
                        entry.pop(field, None)
        abilities, spent = apply_ability_event(before_abilities, used_ability, rest)
        if used_ability and not spent:
            raise StateUnavailable(f"Умение «{used_ability}» сейчас недоступно.")
        current["abilities"] = abilities

    purchase = new_state.pop("purchase", None)
    if purchase and not new_character:
        if not _purchase(current, purchase, latest):
            raise StateUnavailable("Покупку не удалось подтвердить по ответу мастера.")
        for field in ("inventory", "inventory_remove", "gold", "silver", "copper"):
            new_state.pop(field, None)
    resources = new_state.pop("resource_events", {})
    resources = resources if isinstance(resources, dict) else {}
    if not new_character:
        for group, fields in (
            ("inventory", ("inventory", "inventory_remove")),
            ("wallet", ("gold", "silver", "copper")),
        ):
            if not _grounded_event(resources.get(group), latest):
                for field in fields:
                    new_state.pop(field, None)
    inventory = current.get("inventory")
    if isinstance(inventory, list):
        current["inventory"] = _remove_inventory_items(inventory, new_state.get("inventory_remove"))
    merged = dict(current)
    merged["pending_check"] = next_check
    for k, v in new_state.items():
        if v is not None and k not in {"inventory_remove", "level_upgrade", "level_history", "hp_delta", "hp_event",
                                          "quests", "quest_xp_awards", "quest_xp_paid_ids", "last_state_event_id"}:
            merged[k] = v
    if isinstance(current.get("stats"), dict) and isinstance(new_state.get("stats"), dict):
        merged["stats"] = {**current["stats"], **new_state["stats"]}
    old_quests = current.get("quests")
    quest_changes = new_state.get("quests")
    merged["quests"] = _merge_quests(
        old_quests if isinstance(old_quests, list) else [],
        quest_changes if isinstance(quest_changes, list) else [],
    )
    # Preserve arrays when the model accidentally returns empty/missing arrays.
    for f in ("inventory", "notes", "conditions", "relationships", "faction_clocks",
              "dc_modifier_reasons", "proficiencies", "skills", "abilities"):
        if f not in new_state:
            continue
        if f in ("conditions", "dc_modifier_reasons") and isinstance(new_state[f], list):
            merged[f] = new_state[f]
            continue
        old = current.get(f, [])
        new_items = merged.get(f, [])
        if not isinstance(old, list): old = []
        if not isinstance(new_items, list): new_items = old
        if not new_items and old:
            merged[f] = old
        elif old and new_items:
            # Keep existing entries omitted by a model response. Identity by id/name/title,
            # otherwise stable JSON; this conservative merge avoids accidental data loss.
            def ident(item):
                if f == "inventory" and _item_name(item):
                    return _item_name(item)
                if isinstance(item, dict):
                    return item.get("id") or item.get("name") or item.get("title") or json.dumps(item, sort_keys=True, ensure_ascii=False)
                return str(item)
            old_by_id = {ident(item): item for item in old}
            seen = {ident(item) for item in new_items}
            if f in ("notes", "skills", "abilities"):
                updated_items = []
                for item in new_items:
                    previous = old_by_id.get(ident(item))
                    if isinstance(item, dict) and isinstance(previous, dict):
                        item = {**previous, **item}
                    updated_items.append(item)
                new_items = updated_items
            merged[f] = list(new_items) + [item for item in old if ident(item) not in seen]
    # Server-authoritative XP: only explicit new rewards, never double-counted.
    xp = max(0, int(current.get("xp", 0) or 0))
    processed = [event for event in current["xp_event_ids"] if isinstance(event, str)]
    paid = current.get("quest_xp_awards")
    paid = {name for name in paid if isinstance(name, str)} if isinstance(paid, list) else set()
    paid_ids = current.get("quest_xp_paid_ids")
    paid_ids = {value for value in paid_ids if isinstance(value, str)} if isinstance(paid_ids, list) else set()
    quests_by_name = {_quest_name(quest): quest for quest in merged["quests"]}
    paid_ids.update(quest["id"] for name, quest in quests_by_name.items()
                    if name in paid and isinstance(quest.get("id"), str))
    old_completed = {_quest_name(quest) for quest in old_quests or []
                     if isinstance(quest, dict) and quest.get("status") == "completed"}
    newly_completed = {
        name: quest for name, quest in quests_by_name.items()
        if name and name not in old_completed and quest.get("status") == "completed" and not _quest_steps_open(quest)
    }
    for event_id, awards in fresh_events:
        for amount, name in awards:
            if not name and event_id == last_event_id and len(newly_completed) == 1:
                candidate, quest = next(iter(newly_completed.items()))
                reward = quest.get("rewards")
                if isinstance(reward, dict) and reward.get("xp") == amount:
                    name = candidate
            if name:
                quest = quests_by_name.get(name)
                if (name in paid or not quest or str(quest.get("id", "")) in paid_ids
                        or quest.get("status") != "completed"
                        or _quest_steps_open(quest)):
                    continue
                reward = quest.get("rewards")
                promised = reward.get("xp") if isinstance(reward, dict) else None
                if type(promised) is int and promised > 0:
                    amount = promised
                paid.add(name)
                if isinstance(quest.get("id"), str):
                    paid_ids.add(quest["id"])
            xp += amount
        processed.append(event_id)
    for name, quest in newly_completed.items():
        reward = quest.get("rewards")
        promised = reward.get("xp") if isinstance(reward, dict) else None
        if (name not in paid and str(quest.get("id", "")) not in paid_ids
                and type(promised) is int and promised > 0):
            xp += promised
            paid.add(name)
            if isinstance(quest.get("id"), str):
                paid_ids.add(quest["id"])
    merged["quest_xp_awards"] = sorted(paid)
    merged["quest_xp_paid_ids"] = sorted(paid_ids)
    if last_event_id:
        merged["last_state_event_id"] = last_event_id
    merged["xp"] = xp
    merged["xp_event_ids"] = processed
    # Protect core scalar values from blank/null model output.
    for f in ("name", "race", "class", "background", "position"):
        if not merged.get(f) and current.get(f):
            merged[f] = current[f]
    if not new_character:
        # A status line can describe old damage. Only apply a delta tied to the new narration.
        before = int(current.get("hp", 0) or 0)
        delta = new_state.get("hp_delta")
        evidence = new_state.get("hp_event")
        applied = rules_result(current, result)
        grounded = (_grounded_event(evidence, latest) and not any(
            evidence.strip() in message.get("content", "") for message in history[-8:]
            if message.get("role") == "assistant"
        ))
        if applied:
            merged["hp"] = applied["hp_after"]
        else:
            merged["hp"] = before + delta if type(delta) is int and grounded else before
        merged["max_hp"] = current.get("max_hp", DEFAULT_STATE["max_hp"])
    merged["hp"] = max(0, min(int(merged.get("hp", 0) or 0), int(merged.get("max_hp", 1) or 1)))
    previous_level = max(1, int(current.get("level", 1) or 1))
    derived_level = level_for_xp(merged["xp"])
    merged["level"] = derived_level
    merged["xp_next"] = xp_threshold_for_level(derived_level)
    confirmation = messages[-1].get("content", "") if messages and messages[-1].get("role") == "assistant" else ""
    confirmed_levels = {int(level) for level in LEVEL_UPGRADE_RE.findall(confirmation)}
    applied_upgrade = (next_upgrade_level in confirmed_levels
                       and _apply_level_upgrade(merged, new_state.get("level_upgrade"), next_upgrade_level))
    pending = (previous_level - next_upgrade_level + 1) if next_upgrade_level is not None else 0
    merged["level_pending"] = max(0, pending + max(0, derived_level - previous_level) - int(applied_upgrade))
    for currency in ("gold", "silver", "copper"):
        amount = merged.get(currency)
        if type(amount) is not int:
            amount = current.get(currency, 0)
        merged[currency] = max(0, amount) if type(amount) is int else 0
    if str(merged.get("cover", "")).strip().casefold() in {"", "без укрытия", "нет укрытия"}:
        merged["cover"] = "нет"
    merged["dc_modifier"] = int(merged.get("dc_modifier", 0) or 0)
    for f in ("race", "class", "background", "position", "cover", "weapon_in_hand"):
        if not isinstance(merged.get(f), str): merged[f] = current.get(f, DEFAULT_STATE[f])
    if not isinstance(merged.get("restrained"), bool): merged["restrained"] = bool(current.get("restrained", False))
    for f in ("inventory", "notes", "quests", "conditions", "relationships", "faction_clocks",
              "dc_modifier_reasons", "proficiencies", "skills", "abilities", "level_history", "xp_event_ids"):
        if not isinstance(merged.get(f), list): merged[f] = current.get(f, [])
    merged["notes"] = [n for n in (normalize_note(x) for x in merged["notes"]) if n]
    if new_character:
        merged.pop("character_setup_pending", None)
    return merged


# ─── Сохранения ────────────────────────────────────────────────────────────
def _safe_name(raw):
    return re.sub(r"[^a-zA-Z0-9_\-]", "", raw or "")


def _save_path(cid, name):
    """Файл сохранения конкретного игрока: <clientId>~<имя>.json"""
    return SAVES_DIR / f"{cid}~{name}.json"


def _save_backup(path):
    return path.with_name(path.stem + ".bak.json")


def _save_candidates(cid, name):
    """Сначала своё сохранение, потом общее (старые файлы без префикса)."""
    return [_save_path(cid, name), SAVES_DIR / f"{name}.json"]


def _collect_saves(cid):
    """Имя → путь. Свои сохранения перекрывают одноимённые общие."""
    prefix = f"{cid}~"
    own, shared = {}, {}
    for p in SAVES_DIR.glob("*.json"):
        if p.name.endswith(".bak.json"):
            continue
        stem = p.stem
        if stem.startswith(prefix):
            own[stem[len(prefix):]] = p
        elif "~" not in stem:
            shared[stem] = p
    merged = dict(shared)
    merged.update(own)
    return merged


def _save_info(name, path):
    """Имя, время последней записи (когда сохранили) и размер файла."""
    try:
        st = path.stat()
        return {"name": name, "saved_at": st.st_mtime, "size": st.st_size}
    except OSError:
        return {"name": name, "saved_at": 0, "size": 0}


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "app_revision": APP_REVISION,
        "qwen_output_limits": {
            "summary": QWEN_SUMMARY_LIMIT if _is_yandex_qwen(SUMMARY_MODEL) else None,
            "state": QWEN_STATE_LIMIT if _is_yandex_qwen(EXTRACT_MODEL) else None,
            "state_setup": QWEN_STATE_SETUP_LIMIT if _is_yandex_qwen(EXTRACT_MODEL) else None,
        },
        "narration_max_tokens": NARRATION_MAX_TOKENS,
        "narration_setup_tokens": NARRATION_SETUP_TOKENS,
        "narration_retry_tokens": NARRATION_RETRY_TOKENS,
        "recent_history_chars": RECENT_HISTORY_CHARS,
        "summary_input_chars": SUMMARY_INPUT_CHARS,
        "summary_max_tokens": SUMMARY_MAX_TOKENS,
        "frequency_penalty": FREQ_PENALTY,
        "presence_penalty": PRES_PENALTY,
        "provider": PROVIDER,
        "model": MODEL,
        "extract_model": EXTRACT_MODEL,
        "summary_model": SUMMARY_MODEL,
        "base_url_configured": bool(BASE_URL),
        "llm_max_concurrent": LLM_MAX_CONCURRENT,
        "llm_queue_timeout": LLM_QUEUE_TIMEOUT,
        "llm_timeout": LLM_TIMEOUT,
        "saves_dir": str(SAVES_DIR),
        "sessions_cached": len(_SUMMARY_STATE),
    })


@app.route("/api/diag")
def api_diag():
    """Диагностика: что настроено и что реально отвечает провайдер."""
    result = {
        "provider": PROVIDER,
        "model": MODEL,
        "extract_model": EXTRACT_MODEL,
        "summary_model": SUMMARY_MODEL,
        "narration_reasoning": _REASONING_STATE["fix"] if _REASONING_STATE["fix"] is not None else CHAT_REASONING_EFFORT,
        "base_url": BASE_URL or "https://api.openai.com/v1",
        "key_set": bool(API_KEY),
        "env_file": _ENV_FILE.name if _ENV_FILE else None,
        "llm_timeout": LLM_TIMEOUT,
        "saves_dir": str(SAVES_DIR),
    }
    if BASE_URL and ("127.0.0.1" in BASE_URL or "localhost" in BASE_URL):
        result["hint"] = ("base_url указывает на localhost — Ollama видна только с той же "
                          "машины. На Railway так не работает: нужен облачный провайдер "
                          "(например https://openrouter.ai/api/v1).")
    try:
        result["models"] = [m.id for m in client.models.list().data][:60]
        result["model_present"] = MODEL in result["models"]
    except Exception as exc:
        result["models_error"] = _short_error(exc)
    try:
        with llm_slot(20) as ok:
            if not ok:
                raise RuntimeError("нет свободного слота модели")
            resp = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=256 if USE_YANDEX else 8,
                messages=[{"role": "user", "content": "Ответь одним словом: ок"}])
        result["ping"] = (resp.choices[0].message.content or "").strip()
        result["ping_ok"] = bool(result["ping"])
        if not result["ping_ok"]:
            result["ping_error"] = "Модель вернула пустой ответ."
    except Exception as exc:
        result["ping_ok"] = False
        result["ping_error"] = _short_error(exc)
        hint = _error_hint(exc)
        if hint:
            result["ping_hint"] = hint
    return jsonify(result)


@app.route("/")
def index():
    visitor_id, presence = _admit_visitor()
    if not presence["admitted"]:
        return _presence_response(render_template("waiting.html", presence=presence), visitor_id, 503)
    return _presence_response(render_template("index.html", presence=presence), visitor_id)


@app.route("/api/presence", methods=["POST"])
def api_presence():
    visitor_id, presence = _admit_visitor()
    return _presence_response(jsonify(presence), visitor_id)


@app.route("/api/roll", methods=["POST"])
def api_roll():
    data = request.get_json(silent=True) or {}
    try:
        state, messages = data.get("state") or {}, data.get("messages") or []
        if not isinstance(state, dict) or not isinstance(messages, list):
            raise ValueError("Не удалось прочитать состояние проверки.")
        check = active_check(state, messages)
        expression = data.get("expr") or (check["expr"] if check else "1d20")
        if check and formula_key(expression) != formula_key(check["expr"]):
            raise ValueError(f"Ожидается бросок {check['expr']}. Сначала разрешите проверку.")
        roll = roll_dice(expression)
        roll["check_id"] = check["id"] if check else None
        result = roll_result(check, roll)
        result["receipt"] = ROLL_SIGNER.dumps(roll)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/chat", methods=["POST"])
def api_chat():
    started_at = perf_counter()
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    game_state = data.get("state") or DEFAULT_STATE
    if not isinstance(messages, list) or not messages or any(
        not isinstance(message, dict) or not isinstance(message.get("content"), str) for message in messages
    ):
        return jsonify({"error": "История хода повреждена."}), 400
    if not isinstance(game_state, dict):
        game_state = DEFAULT_STATE
    setup_turn = bool(game_state.get("character_setup_pending")
                      or not (game_state.get("race") and game_state.get("class")))
    if messages[0].get("role") != "system":
        prompt = SYSTEM_PROMPT
        if setup_turn:
            prompt += CHARACTER_CREATION_RULES
        messages = [{"role": "system", "content": prompt}] + messages
    action = messages[-1]
    history = messages[:-1]
    check = active_check(game_state, history)
    try:
        result = _signed_roll(action, game_state, history)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    state_context = {key: game_state[key] for key in STATE_FIELDS if key in game_state}
    query = str(action.get("content", ""))
    scene = game_state.get("scene")
    state_context["scene"] = scene if isinstance(scene, dict) else {}
    lookup = " ".join((query, check.get("title", "") if check else "",
                       str(state_context["scene"].get("goal", ""))))
    state_context["facts"] = facts_context(game_state.get("facts"), lookup)
    state_context["journal"] = notes_context(game_state.get("notes"), lookup)
    if result:
        state_context["roll_result"] = result
        state_context["rule_effect"] = rules_result(game_state, result)
    elif check:
        state_context["pending_check"] = check
    if not result and query.startswith("[[БРОСОК]]"):
        return jsonify({"error": "Используйте команду /roll или кнопку проверки."}), 409
    next_upgrade_level = _next_upgrade_level(game_state)
    if next_upgrade_level is not None:
        state_context["next_upgrade_level"] = next_upgrade_level
    quest_context = _quest_context(game_state.get("quests"), game_state.get("quest_xp_awards") or [])
    if quest_context:
        state_context["quests"] = quest_context
    abilities = normalize_abilities(game_state.get("abilities"))
    if abilities:
        known_abilities = []
        descriptions_left = 720
        for ability in abilities:
            if not isinstance(ability, dict) or not isinstance(ability.get("name"), str):
                continue
            entry = {key: ability[key] for key in ("name", "uses_left", "uses_max", "recharge") if key in ability}
            description = ability.get("description")
            if isinstance(description, str) and description.strip() and descriptions_left:
                if (ability["name"].casefold() == "второе дыхание"
                        and "воин" in str(game_state.get("class", "")).casefold()):
                    description = f"Восстановить 1d10+{game_state.get('level', 1)} HP."
                excerpt = description.strip()[:min(180, descriptions_left)]
                entry["description"] = excerpt
                descriptions_left -= len(excerpt)
            known_abilities.append(entry)
        state_context["abilities"] = known_abilities
    inventory = game_state.get("inventory")
    if isinstance(inventory, list):
        state_context["inventory"] = [
            {key: item[key] for key in ("name", "quantity") if key in item}
            if isinstance(item, dict) else item for item in inventory
        ]
    skills = game_state.get("skills")
    if isinstance(skills, list):
        skill_bonuses = {
            skill["name"]: skill["bonus"]
            for skill in skills
            if isinstance(skill, dict) and isinstance(skill.get("name"), str)
            and isinstance(skill.get("bonus"), int)
        }
        if skill_bonuses:
            state_context["skill_bonuses"] = skill_bonuses
    if setup_turn:
        # Пустые числа заготовки только сбивали модель: на первом ходу она их копировала.
        for field in ("hp", "max_hp", "gold", "silver", "copper", "level", "xp", "xp_next", "stats"):
            state_context.pop(field, None)
        if state_context.get("name") == DEFAULT_STATE["name"]:
            state_context.pop("name")
    # При создании героя структуру ответа задаёт CHARACTER_CREATION_RULES,
    # поэтому блок двух абзацев не подмешиваем.
    turn_rule = ("Используй roll_result как окончательный результат именно указанной проверки. "
                 "Если указан rule_effect, используй рассчитанный HP и не применяй изменение повторно. "
                 "Свободный бросок без check не разрешает действие. При pending_check сохраняй "
                 "ожидание, пока игрок не бросит или явно не отменит действие. "
                 "scene — актуальная локация и участники, facts и journal — канон с источниками.")
    length_note = "" if setup_turn else "\n\n" + turn_rule + "\n\n" + LENGTH_NOTE
    state_note = STATE_NOTE_SETUP if setup_turn else STATE_NOTE
    messages = messages[:1] + [{
        "role": "system",
        "content": (state_note + json.dumps(state_context, ensure_ascii=False) + length_note),
    }] + messages[1:]
    turn_id = uuid4().hex
    messages = _messages_with_memory(
        messages, _header_token("X-Session-Id", 64), character_setup=setup_turn,
    )

    visitor_id = request.cookies.get(PRESENCE_COOKIE, "")
    narration_limit = NARRATION_SETUP_TOKENS if setup_turn else NARRATION_MAX_TOKENS

    def generate():
        if not _check_presence(visitor_id, start_chat=True)["admitted"]:
            yield "\n[Ошибка API: место в игре пока недоступно.]"
            return
        queued_at = perf_counter()
        if not LLM_SEM.acquire(timeout=LLM_QUEUE_TIMEOUT):
            _finish_chat_presence(visitor_id)
            yield ("\n[Ошибка API: сервер занят — все слоты модели заняты. "
                   "Подождите минуту и повторите.]")
            return
        slot_at = perf_counter()
        first_text_at = None
        last_usage = None
        attempts = 0
        emitted = False
        try:
            if setup_turn:
                # Создание героя отдаём целиком: обрыв на середине листа показал бы
                # половину карточки, а «дописать» уже отправленный текст нельзя.
                for limit, effort in ((NARRATION_SETUP_TOKENS, CHAT_REASONING_EFFORT),
                                      (NARRATION_RETRY_TOKENS, NARRATION_RETRY_EFFORT)):
                    attempts += 1
                    diag = {}
                    response = None
                    text = ""
                    try:
                        response = _start_completion(messages, max_tokens=limit,
                                                     reasoning_effort=effort, diag=diag)
                    except Exception as e:
                        app.logger.exception("Модель недоступна")
                        hint = _error_hint(e)
                        yield f"\n[Ошибка API: {_short_error(e)}" + (f" {hint}" if hint else "") + "]"
                        return
                    choice = response.choices[0] if getattr(response, "choices", None) else None
                    message = getattr(choice, "message", None)
                    text = (getattr(message, "content", "") or "").strip()
                    finish = getattr(choice, "finish_reason", None)
                    usage = getattr(response, "usage", None)
                    last_usage = usage or last_usage
                    TOKEN_USAGE.record(turn_id, "chat",
                                       getattr(response, "model", None) or MODEL, usage)
                    if diag.get("reasoning_rejected"):
                        _REASONING_STATE["fix"] = ""
                    if text and finish != "length" and sheet_identity(text):
                        if attempts > 1 and not diag.get("reasoning_rejected"):
                            _REASONING_STATE["fix"] = effort or ""
                        emitted = True
                        yield text
                        yield TURN_COMPLETE
                        return
                    print(
                        f"[chat] неполный лист персонажа: попытка={attempts} limit={limit} "
                        f"effort={effort or '—'} finish={finish} chars={len(text)} "
                        f"reasoning_tokens={getattr(getattr(usage, 'completion_tokens_details', None), 'reasoning_tokens', None)}",
                        flush=True,
                    )
                yield "\n[Ошибка API: мастер не вернул полный лист персонажа. Повторите тот же ход.]"
                return
            learned = _REASONING_STATE["fix"]
            variants = [
                (narration_limit, CHAT_REASONING_EFFORT if learned is None else learned),
                # Повтор: лимит выше, рассуждения выключены. Провайдер может
                # отклонить режим — тогда останется хотя бы увеличенный лимит.
                (NARRATION_RETRY_TOKENS, NARRATION_RETRY_EFFORT),
            ]
            for limit, effort in variants:
                attempts += 1
                diag = {}
                stream = None
                usage = None
                model = MODEL
                content_chars = 0
                reasoning_chars = 0
                finish = None
                try:
                    try:
                        stream = _start_stream(messages, max_tokens=limit,
                                               reasoning_effort=effort, diag=diag)
                    except Exception as e:
                        app.logger.exception("Модель недоступна")
                        hint = _error_hint(e)
                        yield f"\n[Ошибка API: {_short_error(e)}" + (f" {hint}" if hint else "") + "]"
                        return
                    for chunk in stream:
                        if getattr(chunk, "usage", None) is not None:
                            usage = chunk.usage
                            model = getattr(chunk, "model", None) or model
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        if getattr(choice, "finish_reason", None):
                            finish = choice.finish_reason
                        delta = getattr(choice, "delta", None)
                        text = getattr(delta, "content", None) if delta is not None else None
                        if not text:
                            # Некоторые прокси игнорируют stream=true и присылают
                            # готовый ответ целиком, без дельт.
                            message = getattr(choice, "message", None)
                            text = getattr(message, "content", None) if message is not None else None
                        if text:
                            content_chars += len(text)
                            if first_text_at is None:
                                first_text_at = perf_counter()
                            emitted = True
                            yield text
                        elif delta is not None:
                            reasoning = (getattr(delta, "reasoning_content", None)
                                         or getattr(delta, "reasoning", None))
                            if reasoning:
                                reasoning_chars += len(reasoning)
                except Exception as e:
                    app.logger.exception("Стрим прервался")
                    if emitted:
                        hint = _error_hint(e)
                        yield f"\n[Ошибка API: {_short_error(e)}" + (f" {hint}" if hint else "") + "]"
                        return
                    # До первого текста ничего не отправлено — повтор безопасен.
                finally:
                    try:
                        if stream is not None:
                            stream.close()
                    finally:
                        if usage is not None:
                            last_usage = usage
                        TOKEN_USAGE.record(turn_id, "chat", model, usage)
                if emitted and finish != "stop":
                    yield "\n[Ошибка API: ответ мастера оборвался. Ход можно повторить.]"
                    return
                if diag.get("reasoning_rejected"):
                    _REASONING_STATE["fix"] = ""
                if emitted:
                    if not diag.get("reasoning_rejected") and attempts > 1:
                        # Запоминаем рабочие параметры, чтобы не платить за повтор каждый ход.
                        _REASONING_STATE["fix"] = effort or ""
                    yield TURN_COMPLETE
                    return
                print(
                    f"[chat] пустой ответ: попытка={attempts} limit={limit} "
                    f"effort={effort or '—'} finish={finish} content_chars={content_chars} "
                    f"reasoning_chars={reasoning_chars} "
                    f"reasoning_tokens={getattr(getattr(usage, 'completion_tokens_details', None), 'reasoning_tokens', None)}",
                    flush=True,
                )
            if not emitted:
                yield ("\n[Ошибка API: модель не вернула текст. Обычно это значит, что "
                       "весь лимит вывода ушёл на скрытые рассуждения. Отправьте сообщение "
                       "ещё раз; если повторяется — задайте в Railway CHAT_REASONING_EFFORT=none "
                       "и перезапустите сервис.]")
        finally:
            LLM_SEM.release()
            _finish_chat_presence(visitor_id)
            print(
                f"[chat] queue_ms={round((slot_at - queued_at) * 1000)} "
                f"first_text_ms={round((first_text_at - started_at) * 1000) if first_text_at is not None else '-'} "
                f"response_ms={round((perf_counter() - started_at) * 1000)} "
                f"attempts={attempts} output_tokens={getattr(last_usage, 'completion_tokens', '-')}",
                flush=True,
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Turn-Id": turn_id},
    )


@app.route("/api/state", methods=["POST"])
def api_state():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    state = data.get("state") or DEFAULT_STATE
    if (not isinstance(state, dict) or not isinstance(messages, list) or not messages
            or any(not isinstance(message, dict) or not isinstance(message.get("content"), str)
                   for message in messages)):
        return jsonify({"error": "История хода или лист повреждены."}), 400
    turn_id = state_turn_id(request.headers.get("X-Turn-Id"))
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return jsonify({"error": "У хода отсутствует завершённый ответ мастера."}), 400
    last_id = message_event_id(last, len(messages) - 1)
    if state.get("last_state_event_id") == last_id:
        return jsonify(state)
    try:
        action = messages[-2] if len(messages) > 1 and messages[-2].get("role") == "user" else None
        result = _signed_roll(action, state, messages[:-2])
        updated = extract_state(messages, state, turn_id, result)
    except StateUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    _schedule_summary(messages, _header_token("X-Session-Id", 64), turn_id)
    return jsonify(updated)


@app.route("/api/saves")
def api_saves_list():
    """Список сохранений игрока: имя, время записи, размер. Новые сверху."""
    saves = _collect_saves(_client_id())
    items = [_save_info(name, path) for name, path in saves.items()]
    items.sort(key=lambda x: x["saved_at"], reverse=True)
    return jsonify(items)


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(silent=True) or {}
    name = _safe_name(data.get("name")) or "default"
    path = _save_path(_client_id(), name)
    payload = {"schema_version": 3, "messages": data.get("messages") or [],
               "state": data.get("state") or DEFAULT_STATE}
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
    data = request.get_json(silent=True) or {}
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
    data = request.get_json(silent=True) or {}
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