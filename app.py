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
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from flask import Flask, request, jsonify, Response, render_template, stream_with_context
from openai import OpenAI

from token_usage import TokenUsage, state_turn_id

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
SUMMARY_MODEL = (os.getenv("SUMMARY_MODEL") or "").strip() or MODEL

print(
    f"[boot] env={_ENV_FILE.name if _ENV_FILE else 'нет (только переменные окружения)'} | "
    f"model={MODEL} | base_url={BASE_URL or 'https://api.openai.com/v1'} | "
    f"key={API_KEY[:4]}… | timeout={LLM_TIMEOUT:g}s",
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
    """Поддерживает провайдеры, отклоняющие штрафы или потоковый учёт токенов."""
    options = dict(model=MODEL, messages=messages, temperature=0.9, stream=True,
                   stream_options={"include_usage": True},
                   frequency_penalty=FREQ_PENALTY, presence_penalty=PRES_PENALTY)
    while True:
        try:
            return client.chat.completions.create(**options)
        except Exception as exc:
            detail = str(exc).lower()
            if ("stream_options" in options
                    and getattr(exc, "status_code", None) in (400, 422, 501)
                    and any(word in detail for word in ("stream_options", "include_usage"))):
                del options["stream_options"]
            elif "frequency_penalty" in options and any(word in detail for word in PENALTY_REJECT):
                del options["frequency_penalty"]
                del options["presence_penalty"]
            else:
                raise


def _measured_completion(turn_id, stage, **options):
    response = None
    try:
        response = client.chat.completions.create(**options)
        return response
    finally:
        if turn_id:
            model = getattr(response, "model", None) or options["model"]
            TOKEN_USAGE.record(turn_id, stage, model, getattr(response, "usage", None))


def _header_token(name, limit):
    """Безопасное значение заголовка: только буквы, цифры, дефис и подчёркивание."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", request.headers.get(name, ""))[:limit]


def _client_id():
    """Идентификатор устройства игрока. Разделяет сохранения между людьми."""
    return _header_token("X-Client-Id", 32) or "shared"


SYSTEM_PROMPT = """
Ты — ведущий текстовой RPG (тёмное фэнтези, D&D, LitRPG). Пиши естественным русским. Веди живой мир, NPC и последствия поступков. Действия, мысли и реплики героя оставляй игроку.

Повествование: покажи результат действия, реакцию мира и голос NPC, когда уместно. Закончи конкретным вызовом: новой ситуацией, поступком NPC или запросом броска. Темп и длина по сцене: от краткого бытового ответа до развёрнутой кульминации. Конкретика важнее эпитетов: «сосны стоят так плотно, что снег не падает на землю». Передавай сцену через несколько органов чувств без перегруза. Меняй лексику, образы и ритм. Не повторяй одну и ту же заметную деталь в соседних сценах и не пересказывай только что сделанный ход. Избегай мета-комментариев и клише вроде «Что вы будете делать?», «Ваше приключение начинается», «судьба зовёт».
У важных NPC есть имя, роль, яркая деталь, характерная речь, цель и тайна. Они помнят поступки, могут лгать и раскрывают сведения постепенно. Веди за кулисами 1–4 силы с целью, методом и часами 1–6: действия игрока и время двигают часы, крупная сцена +1, на 6 цель сбывается. Даже на привале героя мир действует. Показывай последствия в мире, сами часы держи за кадром. Сверяй имена, квесты, места и детали сцены с каноном партии.

XP: каждый результат оплачивай в том же ходу отдельной строкой «+N XP». За победу над врагом CR 1/8–1/4: 10–25; CR 1/2–1: 50–100; CR 2–4: 150–250; CR 5–8: 500–1000; CR 9–12: 1500–2500; CR 13+: 3000–5000. За нелетальную победу 50%, за нескольких врагов суммируй. За решение задачи вне боя: простая 10–30, средняя 50–100, сложная 150–250, эпическая 500+. За ключевую информацию 10–20. За бездействие, разговор без результата и повторную победу над тем же врагом XP нет.
Квесты: промежуточный этап 20–30% общего XP. Все этапы выполнены → завершение и оставшийся XP, основной квест ×2. Завершённый квест больше не меняется, следующий в цепочке активируется сразу. Награды: main 200–500 XP, 100–500 золота и предмет; side 50–150 XP, 20–100 золота; personal 80–200 XP и +1 к характеристике; faction 100–250 XP и репутация.
Уровень и порог XP рассчитывает сервер (100/250/500/1000/2000/3500/5500/8000/12000, затем +5000 за уровень). При повышении предложи выбрать характеристику и новое или улучшенное умение. Применяй бонусы и HP после решения игрока.
Заметки: письма, документы, карты, дневники, слухи и уведомления отправляй в notes, суть сообщай в сцене без полной цитаты. Оружие, броня, зелья, ключи и другие физические предметы идут в inventory.

Броски: рутинные, подготовленные действия и действия без риска или сопротивления описывай сразу. Обычно DC 8–12, шкала 5/8/10/12/15/18/20/25/30. Характеристика 16+ вместе с навыком снижает DC на 2–3. Лёжа +3 в ближнем бою, подъём DC 10 или +5 если позиция мешает; плавание/висение +3, схвачен +5, обездвижен — автопровал, освобождение Сила DC 15. Связан: Ловкость рук DC 15 или Сила DC 20. Укрытие цели +2/+5/полное блокирует. Занятые руки +2, темнота +3, слепота +5, ранение +2 (+3 критическое), яд/страх/спешка +2, местность +1–2. Союзник −2, подготовка −2–3, знакомое −1–2. Назови «1d20 + модификатор характеристики + бонус навыка» и DC с причинами. Результат: 1 критпровал, 2–9 провал с ценой, 10–14 успех с осложнением, 15–19 чистый успех, 20 критуспех.
Навык = модификатор характеристики + мастерство (+2 на 1–4 ур., +3 на 5–8, +4 на 9–12, +5 на 13–16, +6 на 17–20). Сила: Атлетика, Запугивание; Лов: Акробатика, Ловкость рук, Скрытность; Инт: История, Магия, Расследование, Природа, Религия; Мудр: Внимательность, Проницательность, Медицина, Выживание, Животные; Хар: Обман, Убеждение, Выступление.
"""


CHARACTER_SETUP_PROMPT = """
При создании героя сохрани выбранные имя, расу, класс и предысторию, включая авторские варианты. Если раса, класс или предыстория не указаны, предложи 2–3 варианта. Выставь характеристики по классу и расе, владения, навыки, умения и HP. Объяви лист героя, затем начни сцену.
БАЗОВЫЕ ХАРАКТЕРИСТИКИ
• Воин/Варвар/Паладин: Сила 15, Тел 14
• Плут/Следопыт/Монах: Лов 15
• Маг: Инт 15, Лов 12
• Жрец/Друид: Мудр 15, Тел 12
• Бард/Колдун/Чародей: Хар 15, Лов 12
Остальные 10–12.

РАСОВЫЕ БОНУСЫ
Человек +1 всем; Эльф +2 Лов/+1 Инт; Полуэльф +2 Лов/+1 Хар; Дворф +2 Тел/+1 Сила;
Полурослик +2 Лов/+1 Хар; Гном +2 Инт/+1 Тел; Полуорк +2 Сила/+1 Тел;
Тифлинг +2 Хар/+1 Инт; Драконорождённый +2 Сила/+1 Хар; Гоблин/Кобольд +2 Лов/+1 Тел.

ВЛАДЕНИЯ И УМЕНИЯ ПО КЛАССУ
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

ВЛАДЕНИЯ И НАВЫКИ ПО РАСЕ
Человек: 1 навык + 1 язык.
Эльф: Восприятие, тёмное зрение, длинные/короткие мечи, луки, эльфийский.
Полуэльф: 2 навыка, тёмное зрение, эльфийский.
Дворф: тёмное зрение, боевые топоры/молоты, ремесленные, сопр. яду, дварфийский.
Полурослик: Восприятие, скрытность, удача, полурослицкий.
Гном: тёмное зрение, ремесленные, гномий.
Полуорк: тёмное зрение, Запугивание, выносливость, орочий.
Тифлинг: тёмное зрение, сопр. огню, Адское возмездие (1/долгий), инфернальный.
Драконорождённый: Дыхание дракона (1/короткий, 2d6), сопр. стихии, драконий.

НАВЫКИ ПО ПРЕДЫСТОРИИ
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


STATE_EXTRACT_PROMPT = """
Обнови лист RPG по последнему ходу. Более ранние реплики даются для контекста. Верни только изменившиеся поля валидным JSON без Markdown, либо {}. Не выдумывай событий. XP, уровень, порог и историю начислений ведёт сервер.
Поля: name, race, class, background (меняй только при событии); hp (не выше max_hp), max_hp, gold; stats {str,dex,con,int,wis,cha}; proficiencies (строки, до 12); skills {name,bonus,source} (6–8); abilities {name,description,uses_max,uses_left,recharge} (5–6; passive/short_rest/long_rest/day). Создай владения, навыки и умения при создании героя; учти расход и восстановление зарядов. Если лист ещё пуст, установи базовые характеристики по классу и расе; уже заполненные сохраняй. Применяй повышение характеристик, HP и умение после выбора игрока, запиши level_history {level,hp,stats,abilities}, пересчитай бонусы навыков на 5/9/13/17 уровнях.
inventory — физические предметы, conditions — состояния. Письма, заметки, документы, уведомления, слухи и дневники идут в notes {id,type,title,content,source,date,unread}. type: note/letter/document/notification/rumor/journal. Полный текст или суть в 1–4 предложениях, новые unread=true, известные записи не дублируй. Список известных заметок и завершённых квестов справочный.
quests {id,name,type,status,chain,chain_order,stages:[{text,done}],current_stage,rewards:{xp,gold,items,reputation},giver,location}. type: main/side/personal/faction; status: active/completed/failed. Все этапы закрыты → completed и активация следующего в цепочке. Максимум 8 активных. Для изменённых записей сохраняй id, присылай запись целиком.
relationships {name,attitude,note}; faction_clocks {name,progress,max,goal} (до 4).
dc_modifier и dc_modifier_reasons отражают фактические условия; нейтрально 0 и []. position: стоит/сидит/на коленях/лежит/верхом/плавает/висит/схвачен/обездвижен. cover: нет/половинное/три_четверти/полное. weapon_in_hand: оружие/щит/обе_руки_заняты/пустая строка. restrained: bool.
"""


SUMMARY_PROMPT = """
Объедини старое резюме и новые события в 4–6 плотных предложений. Сохрани сеттинг, важных NPC и их отношение, решения и последствия, открытые цели, долги, улики, активные квесты, актуальное положение героя. Убирай повторы и дословные реплики. При противоречии учитывай последнее событие.
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

# Что уходит модели в промпте чата. Заметки, квесты и история уровней
# остаются в диалоге и в отдельном листе — в каждый запрос их не шлём.
STATE_FIELDS = ("name", "race", "class", "background", "level", "xp", "xp_next",
                "hp", "max_hp", "gold", "stats", "conditions", "position",
                "cover", "weapon_in_hand", "restrained")
STATE_NOTE = ("[АКТУАЛЬНОЕ СОСТОЯНИЕ ПЕРСОНАЖА — справочная информация, "
              "не меняй числа без игрового события]\n")


MAX_MESSAGES_BEFORE_SUMMARY = 18
KEEP_RECENT = 10
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


def summarize_old_messages(messages, session_id="", turn_id=None):
    """Сворачивает старую часть диалога в одно резюме.

    Резюме хранится по session_id. Когда несвёрнутый диалог перерастает окно,
    добавляем в резюме накопившуюся часть и оставляем последние реплики целиком.
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

    tail = dialogue[folded:]
    if len(tail) <= MAX_MESSAGES_BEFORE_SUMMARY:
        return pack(summary_text, tail)

    target = len(dialogue) - KEEP_RECENT
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
            resp = _measured_completion(turn_id, "summary",
                model=SUMMARY_MODEL,
                messages=[{"role": "system", "content": SUMMARY_PROMPT},
                          {"role": "user", "content":
                           f"Материал для обновления памяти:\n{archive_input}"}],
                temperature=0.2,
            )
        new_summary = (resp.choices[0].message.content or "").strip()
        if not new_summary:
            raise RuntimeError("пустое резюме")
    except Exception:
        return pack(summary_text, tail[-MAX_MESSAGES_BEFORE_SUMMARY:])

    if persist:
        _summary_put(session_id, new_summary, target)
    return pack(new_summary, dialogue[target:])


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


def extraction_context(current):
    # Полные тексты заметок и завершённые этапы остаются в сохранении.
    # Модели достаточно известных ID и названий, чтобы не создавать дубликаты.
    context = {key: current[key] for key in DEFAULT_STATE
               if key not in ("xp_event_ids", "xp_next", "notes", "quests", "level_history")}
    notes = current.get("notes")
    if isinstance(notes, list):
        context["known_notes"] = [
            {key: note[key] for key in ("id", "title", "type") if key in note}
            if isinstance(note, dict) else note[:40]
            for note in notes if isinstance(note, (dict, str))
        ]
    quests = current.get("quests")
    if isinstance(quests, list):
        context["quests"] = [
            {key: quest[key] for key in ("id", "name", "status", "chain", "chain_order") if key in quest}
            if isinstance(quest, dict) and quest.get("status") in ("completed", "failed")
            else quest for quest in quests
        ]
    return context


def extract_state(messages, current_state, turn_id=None):
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

    recent = messages[-5:]
    dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in recent if m.get("content"))
    context = json.dumps(extraction_context(current), ensure_ascii=False, separators=(",", ":"))
    prompt = f"ЛИСТ:\n{context}\n\nПОСЛЕДНИЙ ХОД С КОНТЕКСТОМ:\n{dialogue}"
    try:
        with llm_slot() as ok:
            if not ok:
                raise RuntimeError("нет свободного слота модели")
            resp = _measured_completion(turn_id, "state",
                model=EXTRACT_MODEL,
                messages=[{"role": "system", "content": STATE_EXTRACT_PROMPT},
                         {"role": "user", "content": prompt}],
                temperature=0.1,
            )
        new_state = clean_json(resp.choices[0].message.content or "")
    except Exception as exc:
        app.logger.warning("Извлечение состояния не удалось: %s", _short_error(exc))
        new_state = {}

    merged = dict(current)
    for k, v in new_state.items():
        if k in DEFAULT_STATE and v is not None:
            merged[k] = v
    if isinstance(new_state.get("stats"), dict):
        old_stats = current.get("stats")
        merged["stats"] = {**DEFAULT_STATE["stats"],
                           **(old_stats if isinstance(old_stats, dict) else {}),
                           **{key: value for key, value in new_state["stats"].items()
                              if key in DEFAULT_STATE["stats"] and type(value) is int}}
    else:
        merged["stats"] = current["stats"]
    # Preserve arrays when the model accidentally returns empty/missing arrays.
    for f in ("inventory", "notes", "quests", "conditions", "relationships", "faction_clocks",
              "dc_modifier_reasons", "proficiencies", "skills", "abilities", "level_history"):
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
                if isinstance(item, dict):
                    return item.get("id") or item.get("name") or item.get("title") or json.dumps(item, sort_keys=True, ensure_ascii=False)
                return str(item)
            seen = {ident(x) for x in new_items}
            merged[f] = list(new_items) + [x for x in old if ident(x) not in seen]
    # Server-authoritative XP: only explicit new rewards, never double-counted.
    xp = max(0, int(current.get("xp", 0) or 0))
    processed = list(current.get("xp_event_ids", []))
    for event_id, amount in fresh_events:
        if event_id not in processed:
            xp += amount
            processed.append(event_id)
    merged["xp"] = xp
    merged["xp_event_ids"] = processed[-500:]
    # Protect core scalar values from blank/null model output.
    for f in ("name", "race", "class", "background", "position", "cover", "weapon_in_hand"):
        if not merged.get(f) and current.get(f):
            merged[f] = current[f]
    merged["hp"] = max(0, min(int(merged.get("hp", 0) or 0), int(merged.get("max_hp", 1) or 1)))
    previous_level = max(1, int(current.get("level", 1) or 1))
    derived_level = level_for_xp(merged["xp"])
    merged["level"] = derived_level
    merged["xp_next"] = xp_threshold_for_level(derived_level)
    merged["level_pending"] = max(0, derived_level - previous_level)
    merged["gold"] = max(0, int(merged.get("gold", 0) or 0))
    merged["dc_modifier"] = int(merged.get("dc_modifier", 0) or 0)
    for f in ("race", "class", "background", "position", "cover", "weapon_in_hand"):
        if not isinstance(merged.get(f), str): merged[f] = current.get(f, DEFAULT_STATE[f])
    if not isinstance(merged.get("restrained"), bool): merged["restrained"] = bool(current.get("restrained", False))
    for f in ("inventory", "notes", "quests", "conditions", "relationships", "faction_clocks",
              "dc_modifier_reasons", "proficiencies", "skills", "abilities", "level_history", "xp_event_ids"):
        if not isinstance(merged.get(f), list): merged[f] = current.get(f, [])
    merged["quests"] = [q for q in (normalize_quest(x) for x in merged["quests"]) if q]
    merged["notes"] = [n for n in (normalize_note(x) for x in merged["notes"]) if n]
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
        "model": MODEL,
        "extract_model": EXTRACT_MODEL,
        "summary_model": SUMMARY_MODEL,
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
                model=MODEL, temperature=0, max_tokens=8,
                messages=[{"role": "user", "content": "Ответь одним словом: ок"}])
        result["ping_ok"] = True
        result["ping"] = (resp.choices[0].message.content or "").strip()
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
    messages = data.get("messages") or []
    game_state = data.get("state") or DEFAULT_STATE
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "empty messages"}), 400
    if messages[0].get("role") != "system":
        needs_setup = (
            not isinstance(game_state, dict)
            or not game_state.get("race")
            or not game_state.get("class")
            or not game_state.get("background")
            or game_state.get("character_setup_pending")
            or not any(message.get("role") == "assistant" for message in messages)
        )
        system_prompt = SYSTEM_PROMPT + (CHARACTER_SETUP_PROMPT if needs_setup else "")
        messages = [{"role": "system", "content": system_prompt}] + messages
    state_context = {k: v for k, v in (game_state.items() if isinstance(game_state, dict) else [])
                    if k in STATE_FIELDS}
    compact_state = json.dumps(state_context, ensure_ascii=False, separators=(",", ":"))
    messages = messages[:1] + [{"role": "system", "content": STATE_NOTE + compact_state}] + messages[1:]
    turn_id = uuid4().hex
    messages = summarize_old_messages(messages, _header_token("X-Session-Id", 64), turn_id)

    def generate():
        if not LLM_SEM.acquire(timeout=LLM_QUEUE_TIMEOUT):
            yield ("\n[Ошибка API: сервер занят — все слоты модели заняты. "
                   "Подождите минуту и повторите.]")
            return
        stream = None
        usage = None
        model = MODEL
        try:
            try:
                stream = _start_stream(messages)
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
                delta = getattr(chunk.choices[0], "delta", None)
                text = getattr(delta, "content", None) if delta else None
                if text:
                    yield text
        except Exception as e:
            app.logger.exception("Стрим прервался")
            hint = _error_hint(e)
            yield f"\n[Ошибка API: {_short_error(e)}" + (f" {hint}" if hint else "") + "]"
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                LLM_SEM.release()
                TOKEN_USAGE.record(turn_id, "chat", model, usage)

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
    if not isinstance(messages, list) or not messages:
        return jsonify(state)
    turn_id = state_turn_id(request.headers.get("X-Turn-Id"))
    return jsonify(extract_state(messages, state, turn_id))


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
    payload = {"schema_version": 2, "messages": data.get("messages") or [],
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
