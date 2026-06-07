#!/usr/bin/env python3
"""EduHelper Telegram bot.

Features:
- Answers school and student subject questions.
- Explains topics simply with examples.
- Generates quizzes and checks user answers.
- Summarizes textbooks/articles by link or text.
- Tracks user learning progress in SQLite.
"""

import html
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import openai
except ImportError:
    openai = None

DB_PATH = os.path.join(os.path.dirname(__file__), "eduh_helper.db")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
GROQ_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "groq-1")


def load_environment() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        if load_dotenv is not None:
            load_dotenv(dotenv_path=env_path)
            return
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                if key and key not in os.environ:
                    os.environ[key] = value


def create_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            topic TEXT,
            correct INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0,
            last_update TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, topic)
        )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS quizzes (
            quiz_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            topic TEXT,
            quiz_data TEXT,
            current_index INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.commit()
    conn.close()


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def register_user(user_id: int, username: str, first_name: str, last_name: str) -> None:
    conn = get_db_connection()
    conn.execute(
        "INSERT OR IGNORE INTO users (user_id, username, first_name, last_name) VALUES (?, ?, ?, ?)",
        (user_id, username, first_name, last_name),
    )
    conn.commit()
    conn.close()


def update_progress(user_id: int, topic: str, correct: int, total: int) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT correct, total FROM progress WHERE user_id = ? AND topic = ?",
        (user_id, topic),
    )
    row = cursor.fetchone()
    if row:
        cursor.execute(
            "UPDATE progress SET correct = correct + ?, total = total + ?, last_update = CURRENT_TIMESTAMP WHERE user_id = ? AND topic = ?",
            (correct, total, user_id, topic),
        )
    else:
        cursor.execute(
            "INSERT INTO progress (user_id, topic, correct, total) VALUES (?, ?, ?, ?)",
            (user_id, topic, correct, total),
        )
    conn.commit()
    conn.close()


def save_quiz(user_id: int, topic: str, quiz_data: List[Dict[str, Any]]) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO quizzes (user_id, topic, quiz_data, current_index, correct_count) VALUES (?, ?, ?, 0, 0)",
        (user_id, topic, json.dumps(quiz_data, ensure_ascii=False)),
    )
    quiz_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return quiz_id


def load_quiz(quiz_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM quizzes WHERE quiz_id = ?", (quiz_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "quiz_id": row[0],
        "user_id": row[1],
        "topic": row[2],
        "quiz_data": json.loads(row[3]),
        "current_index": row[4],
        "correct_count": row[5],
    }


def update_quiz_progress(quiz_id: int, current_index: int, correct_count: int) -> None:
    conn = get_db_connection()
    conn.execute(
        "UPDATE quizzes SET current_index = ?, correct_count = ? WHERE quiz_id = ?",
        (current_index, correct_count, quiz_id),
    )
    conn.commit()
    conn.close()


def cleanup_quiz(quiz_id: int) -> None:
    conn = get_db_connection()
    conn.execute("DELETE FROM quizzes WHERE quiz_id = ?", (quiz_id,))
    conn.commit()
    conn.close()


def query_openai(messages: List[Dict[str, str]], max_tokens: int = 800) -> str:
    if openai is None:
        raise RuntimeError("OpenAI package not installed. Install it with pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Не найден OPENAI_API_KEY. Установите ключ в окружении или .env.")
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=DEFAULT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        request_timeout=10,
    )
    return response.choices[0].message.content.strip()


def query_groq(prompt: str, max_tokens: int = 800) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    endpoint = os.getenv("GROQ_API_URL", "https://api.groq.com/v1/generate")
    payload = {
        "model": GROQ_DEFAULT_MODEL,
        "prompt": prompt,
        "max_output_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and "output" in data:
        output = data["output"]
        if isinstance(output, list):
            return "\n".join(str(item) for item in output)
        return str(output)
    return str(data)


def ask_language_model(prompt: str, system_prompt: Optional[str] = None) -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if groq_key:
        return query_groq(prompt)
    if openai_key:
        system_prompt = system_prompt or "You are a helpful educational assistant for school and university students."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return query_openai(messages)
    raise RuntimeError(
        "Я не могу ответить: не настроен API-ключ OpenAI или Groq. "
        "Добавьте OPENAI_API_KEY или GROQ_API_KEY в .env или окружении."
    )


def sanitize_topic(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ \-_,.;:()]+", "", text).strip()


def looks_like_url(text: str) -> bool:
    parsed = urlparse(text.strip())
    return bool(parsed.scheme and parsed.netloc)


def extract_text_from_html(html_text: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", html_text, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_text_from_url(url: str) -> str:
    response = requests.get(url, timeout=15, headers={"User-Agent": "EduHelperBot/1.0"})
    response.raise_for_status()
    return extract_text_from_html(response.text)


def parse_quiz_json(raw_text: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # Fallback parsing: look for numbered questions and answer options.
    questions: List[Dict[str, Any]] = []
    blocks = re.split(r"\n\s*\d+[.)]", raw_text)
    for block in blocks[1:]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        question_text = lines[0]
        options = []
        answer_index = 0
        for line in lines[1:]:
            match = re.match(r"^([A-Da-d])[.)]\s*(.*)$", line)
            if match:
                options.append(match.group(2).strip())
        if len(options) >= 2:
            questions.append({
                "question": question_text,
                "options": options,
                "answer": 0,
            })
    return questions


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    register_user(user.id, user.username or "", user.first_name or "", user.last_name or "")
    text = (
        "Привет! Я EduHelper — твой учебный помощник.\n\n"
        "Я могу:"
        "\n• Отвечать на вопросы по школьным предметам"
        "\n• Объяснять темы простым языком"
        "\n• Генерировать тесты и проверять ответы"
        "\n• Делиться краткими summary по ссылкам и тексту"
        "\n• Отслеживать прогресс обучения\n\n"
        "Команды:\n"
        "/ask <вопрос> — задать вопрос\n"
        "/summary <текст или ссылка> — краткое summary\n"
        "/test <тема> — сделать тест\n"
        "/progress — посмотреть прогресс\n"
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start_command(update, context)


async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args).strip()
    if not query:
        if update.message.reply_to_message and update.message.reply_to_message.text:
            query = update.message.reply_to_message.text.strip()
        else:
            await update.message.reply_text("Напиши вопрос после /ask, например: /ask что такое интеграл?")
            return
    prompt = (
        "Объясни на простом языке и приведи пример."
        f"\n\nВопрос: {query}\n"
        "Ответь как учитель-помощник для школьника или студента."
    )
    await update.message.reply_text("Собираю ответ... Подожди немного.")
    try:
        answer = ask_language_model(prompt)
    except Exception as exc:
        logger.exception(exc)
        error_text = str(exc)
        if "API-ключ" in error_text or "OPENAI_API_KEY" in error_text or "GROQ_API_KEY" in error_text:
            await update.message.reply_text(
                "Я не могу ответить сейчас: настройте OPENAI_API_KEY или GROQ_API_KEY в .env или в переменных окружения."
            )
        else:
            await update.message.reply_text(f"Ошибка при обработке запроса: {exc}")
        return
    await update.message.reply_text(answer)


async def handle_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args).strip()
    if not query and update.message.reply_to_message and update.message.reply_to_message.text:
        query = update.message.reply_to_message.text.strip()
    if not query:
        await update.message.reply_text("Отправь текст или ссылку после /summary для краткого summary.")
        return
    await update.message.reply_text("Готовлю краткое summary...")
    try:
        if looks_like_url(query):
            page_text = fetch_text_from_url(query)
            if not page_text:
                raise ValueError("Не удалось получить текст с указанной ссылки.")
            source = page_text[:3000]
            prompt = (
                "Сделай краткое и понятное summary учебного текста."
                " Приведи ключевые идеи и практический вывод."
                f"\n\nТекст:\n{source}"
            )
        else:
            source = query[:3000]
            prompt = (
                "Сделай краткое и понятное summary учебного текста."
                " Приведи ключевые идеи и практический вывод."
                f"\n\nТекст:\n{source}"
            )
        answer = ask_language_model(prompt)
    except Exception as exc:
        logger.exception(exc)
        error_text = str(exc)
        if "API-ключ" in error_text or "OPENAI_API_KEY" in error_text or "GROQ_API_KEY" in error_text:
            await update.message.reply_text(
                "Я не могу сделать summary сейчас: настройте OPENAI_API_KEY или GROQ_API_KEY в .env или в переменных окружения."
            )
        else:
            await update.message.reply_text(f"Ошибка при создании summary: {exc}")
        return
    await update.message.reply_text(answer)


async def handle_test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    topic = " ".join(context.args).strip()
    if not topic and update.message.reply_to_message and update.message.reply_to_message.text:
        topic = update.message.reply_to_message.text.strip()
    if not topic:
        await update.message.reply_text("Напиши тему после /test, например: /test физика скорость.")
        return
    topic = sanitize_topic(topic)
    await update.message.reply_text(f"Генерирую тест по теме: {topic}. Подожди...")
    prompt = (
        "Сгенерируй JSON-массив из 3 вопросов с несколькими вариантами ответа по теме обучения."
        " Каждый объект должен содержать поля question, options и answer (индекс правильного варианта)."
        f"\n\nТема: {topic}\n"
        "Верни только JSON.")
    try:
        raw = ask_language_model(prompt)
        quiz_data = parse_quiz_json(raw)
        if not quiz_data:
            raise ValueError("Модель вернула некорректный формат quiz. Попробуй ещё раз.")
        quiz_id = save_quiz(update.effective_user.id, topic, quiz_data)
        await send_quiz_question(update, context, quiz_id)
    except Exception as exc:
        logger.exception(exc)
        await update.message.reply_text(f"Ошибка при создании теста: {exc}")


async def send_quiz_question(update: Update, context: ContextTypes.DEFAULT_TYPE, quiz_id: int) -> None:
    quiz = load_quiz(quiz_id)
    if not quiz:
        await update.message.reply_text("Не удалось загрузить тест.")
        return
    questions = quiz["quiz_data"]
    index = quiz["current_index"]
    if index >= len(questions):
        await update.message.reply_text("Тест уже завершен.")
        return
    question = questions[index]
    text = f"Вопрос {index + 1}/{len(questions)}:\n{question['question']}"
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(f"{chr(65+i)}. {choice}", callback_data=f"quiz|{quiz_id}|{index}|{i}")]
            for i, choice in enumerate(question["options"])
        ]
    )
    await update.message.reply_text(text, reply_markup=keyboard)


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    parts = query.data.split("|")
    if parts[0] != "quiz" or len(parts) != 4:
        await query.edit_message_text("Неизвестная кнопка.")
        return
    quiz_id = int(parts[1])
    q_index = int(parts[2])
    choice = int(parts[3])
    quiz = load_quiz(quiz_id)
    if not quiz:
        await query.edit_message_text("Этот тест больше не активен.")
        return
    questions = quiz["quiz_data"]
    if q_index >= len(questions):
        await query.edit_message_text("Этот вопрос уже пройден.")
        return
    question = questions[q_index]
    correct_index = int(question.get("answer", 0))
    is_correct = choice == correct_index
    feedback = "✅ Верно!" if is_correct else f"❌ Неверно. Правильный ответ: {chr(65 + correct_index)}"
    new_correct_count = quiz["correct_count"] + (1 if is_correct else 0)
    next_index = quiz["current_index"] + 1
    update_quiz_progress(quiz_id, next_index, new_correct_count)
    update_progress(update.effective_user.id, quiz["topic"], 1 if is_correct else 0, 1)
    if next_index >= len(questions):
        await query.edit_message_text(f"{feedback}\n\nТест завершён. Правильных ответов: {new_correct_count}/{len(questions)}")
        cleanup_quiz(quiz_id)
        return
    await query.edit_message_text(f"{feedback}\n\nСледующий вопрос готов.")
    # Send next question separately because edited text cannot include new keyboard reliably.
    await send_quiz_question(update, context, quiz_id)


async def handle_progress(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT topic, correct, total FROM progress WHERE user_id = ? ORDER BY total DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        await update.message.reply_text("Прогресс пока не зафиксирован. Пройди тест или задай вопрос.")
        return
    lines = ["Твой прогресс по темам:"]
    for row in rows:
        topic = row[0]
        correct = row[1]
        total = row[2]
        percent = int(correct / total * 100) if total else 0
        lines.append(f"• {topic}: {correct}/{total} ({percent}%)")
    await update.message.reply_text("\n".join(lines))


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.strip()
    if not text:
        return
    await update.message.reply_text("Думаю над ответом...")
    try:
        prompt = (
            "Ты — помощник EduHelper. Ответь просто и ясно, как школьнику или студенту."
            f"\n\nВопрос: {text}"
            "\n\nЕсли вопрос касается темы, объясни её и приведи пример."
        )
        answer = ask_language_model(prompt)
    except Exception as exc:
        logger.exception(exc)
        error_text = str(exc)
        if "API-ключ" in error_text or "OPENAI_API_KEY" in error_text or "GROQ_API_KEY" in error_text:
            await update.message.reply_text(
                "Я не могу ответить сейчас: настройте OPENAI_API_KEY или GROQ_API_KEY в .env или в переменных окружения."
            )
        else:
            await update.message.reply_text(f"Ошибка при обработке сообщения: {exc}")
        return
    await update.message.reply_text(answer)


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я не понимаю эту команду. Воспользуйся /help для списка доступных команд."
    )


def main() -> None:
    load_environment()
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN не задан. Установите переменную окружения или добавьте его в .env файл.")
    create_db()
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", handle_ask))
    app.add_handler(CommandHandler("summary", handle_summary))
    app.add_handler(CommandHandler("test", handle_test))
    app.add_handler(CommandHandler("progress", handle_progress))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    logger.info("EduHelper bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
