import argparse
import hashlib
from pathlib import Path
import json
import sys
import uuid

from utils.ollama import embed, embed_chunks
from utils.translation.lang import set_lang, detect_system_lang, get_current_lang, t
from utils.debug import d

# инициализируем глобальную переменную текущего языка
# позже язык будет адаптироваться после запросов пользователя
set_lang(detect_system_lang())

try:
    import chromadb
    import ollama
    import nltk
    from pydantic import BaseModel
    import telebot
    from telebot.types import ReplyKeyboardMarkup, KeyboardButton
except ImportError as e:
    print(t("install_deps"))
    sys.exit(1)

nltk.download("punkt", quiet=True)
from nltk import sent_tokenize

import yake

# классификаторы
from classifiers.language import LanguageClassifier
from classifiers.moral import MoralFilter
from classifiers.domain import DomainClassifier

BOOKS_DIR = Path("./data/books")
RECO_DIR = Path("./data/reco")
TABLE_DIR = Path("./data/table")
COT_DIR = Path("./data/cot")
CHROMA_DB_DIR = "./chroma_db"
BOOKS_COLLECTION = "books"
RECO_COLLECTION = "recommendations"
TABLE_COLLECTION = "tables"
COT_COLLECTION = "reasoning"
DEFAULT_CHAT_MODEL = "qwen2.5:7b"
CHUNK_SENTENCES = 6
TOP_K = 5
TG_BOT_TOKEN = "" # 


# ----------------------------
# Утилиты
# ----------------------------
def file_hash_for_dir(directory: Path) -> str:
    hasher = hashlib.md5()
    for path in sorted(directory.rglob("*.txt")):
        hasher.update(str(path).encode())
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def hash_file_for_collection(collection_name: str) -> Path:
    return Path(f"db_hash_{collection_name}.txt")


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, max_sentences=CHUNK_SENTENCES):
    sents = sent_tokenize(text)
    return [
        " ".join(sents[i : i + max_sentences])
        for i in range(0, len(sents), max_sentences)
    ]


def infer_title_author(fp: Path):
    name = fp.stem
    for sep in [" - ", " — ", "–", "-"]:
        if sep in name:
            author, title = name.split(sep, 1)
            return title.strip(), author.strip()
    return name, "Unknown"


def extract_ollama_content(resp):
    # поддерживаем несколько форматов ответа
    try:
        if isinstance(resp, dict):
            # resp может быть dict с ключом 'message' или 'content'
            if (
                "message" in resp
                and isinstance(resp["message"], dict)
                and "content" in resp["message"]
            ):
                return resp["message"]["content"]
            if "content" in resp:
                return resp["content"]
        # объект с атрибутами
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            return resp.message.content
        if hasattr(resp, "content"):
            return resp.content
    except Exception:
        pass
    return str(resp)


# ----------------------------
# ChromaDB helpers
# ----------------------------
def get_client(path=CHROMA_DB_DIR):
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(client, name):
    return (
        client.get_collection(name)
        if name in [c.name for c in client.list_collections()]
        else client.create_collection(name)
    )
    
def add_text_to_collection(text: str, collection, meta: dict):
    emb = embed(text)
    collection.add(
        documents=[text],
        embeddings=[emb],
        metadatas=[meta],
        ids=[str(uuid.uuid4())],
    )


# ----------------------------
# Построение базы
# ----------------------------
def build_db(dir_path: Path, collection_name, rebuild=False):
    if not dir_path.exists():
        print(t("dir_not_found", dir_path=dir_path))
        return None

    cur_hash = file_hash_for_dir(dir_path)
    hash_file = hash_file_for_collection(collection_name)
    if hash_file.exists():
        old_hash = hash_file.read_text().strip()
    else:
        old_hash = ""

    if not rebuild and cur_hash == old_hash:
        print(t("db_up_to_date", collection_name=collection_name))
        client = get_client()
        return get_or_create_collection(client, collection_name)

    print(t("db_building", collection_name=collection_name))
    client = get_client()
    if collection_name in [c.name for c in client.list_collections()]:
        try:
            client.delete_collection(collection_name)
        except:
            pass
    collection = client.create_collection(collection_name)

    ids, docs, metas = [], [], []
    for fp in sorted(dir_path.rglob("*.txt")):
        text = read_text_file(fp)
        title, author = infer_title_author(fp)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{fp.stem}__{i}")
            docs.append(chunk)
            metas.append(
                {
                    "source_file": fp.name,
                    "title": title,
                    "author": author,
                    "chunk_index": i,
                }
            )

    embeddings = embed_chunks(docs)
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    hash_file.write_text(cur_hash)
    print(t("db_built", collection_name=collection_name))
    return collection


# ----------------------------
# Поиск по базе
# ----------------------------
def retrieve_topk(query, collection, top_k=TOP_K):
    q_emb = embed(query)

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = [
        {"doc": d, "meta": m, "distance": dist}
        for d, m, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
    return hits


def build_prompt(question, hits, max_chars=3000):
    system = t("system_prompt_rag")
    context = ""
    total_len = 0
    for i, h in enumerate(hits, 1):
        title = h['meta'].get('title', 'Unknown')
        author = h['meta'].get('author', 'Unknown')
        block = f"[{i}] {title} — {author}:\n{h['doc']}\n"
        if total_len + len(block) > max_chars:
            context += block[: max_chars - total_len] + t("context_truncated")
            break
        context += block + "\n"
        total_len += len(block)
    user = t("rag_user_prompt", context=context, question=question)
    return {"system": system, "user": user}


# ----------------------------
# Вывод табличных данных
# ----------------------------
class Attraction(BaseModel):
    name: str
    domain: str


class TableAnswer(BaseModel):
    table: list[Attraction]


def extract_table_from_text(text: str, model=DEFAULT_CHAT_MODEL, collection=None):
    """Извлекает объекты/достопримечательности через LLM и возвращает список словарей"""
    choice = None
    if collection:
        print(t("table_choice_prompt"))
        choice = input(t("your_choice")).strip()

        if choice == "1":
            # Получаем top-k релевантных документов
            hits = retrieve_topk(text, collection)
            context = ""
            for i, h in enumerate(hits, 1):
                title = h['meta'].get('title', 'Unknown')
                author = h['meta'].get('author', 'Unknown')
                context += f"[{i}] {title} — {author}:\n{h['doc']}\n\n"
            text = f"Текст из базы:\n{context}\nЗапрос от пользователя: {text}"
        elif choice != "2":
            print(t("invalid_choice") + "\n")
            return []
    
    try:
        resp = ollama.chat(
            messages=[
                {"role": "system", "content": t("table_extract_system")},
                {"role": "user", "content": text},
            ],
            model=model,
            options={"temperature": 0, "top_p": 1},
            format=TableAnswer.model_json_schema(),
        )
        content = extract_ollama_content(resp)
        table = json.loads(content)["table"]
        return table
    except Exception as e:
        print(t("table_extract_error"), e)
        return []


def render_markdown_table(table: list[dict]) -> str:
    """Преобразует список словарей в Markdown-таблицу"""
    md = t("table_md_header")
    for row in table:
        md += f"| {row['name']} | {row['domain']} |\n"
    return md


def render_table_for_tg(table: list[dict]) -> str:
    md = t("table_plain_header")
    md += "----------------------------|----------------\n"
    for row in table:
        md += f"{row['name']:<28} | {row['domain']}\n"
    return f"```\n{md}```"


# ----------------------------
# Решение сложных текстовых задач с шагами
# ----------------------------
def solve_complex_task_with_steps(
    task_text: str, collection=None, model=DEFAULT_CHAT_MODEL, top_k=3
) -> str:
    """
    Решает текстовую задачу с few-shot CoT, используя RAG для подбора примеров.
    """
    if collection:
        hits = retrieve_topk(task_text, collection, top_k=top_k)
        # Строим few-shot контекст из retrieved документов
        few_shot_context = ""
        for i, h in enumerate(hits, 1):
            title = h['meta'].get('title', 'Unknown')
            author = h['meta'].get('author', 'Unknown')
            few_shot_context += f"[{t('cot_example')} {i}] {title} — {author}:\n{h['doc']}\n\n"
    else:
        few_shot_context = ""

    system_msg = t("cot_system")
    user_msg = t("cot_user", few_shot_context=few_shot_context, task_text=task_text)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = ollama.chat(
            messages=messages,
            model=model,
            options={"temperature": 0, "top_p": 1},
        )
        return extract_ollama_content(resp)
    except Exception as e:
        print(t("cot_error"), e)
        return t("cot_fail")

# Добавление в базу
def add_to_db(text, book_col, reco_col, table_col, cot_col):
    USER_COLLECTIONS = {
        "1": book_col,
        "2": reco_col,
        "3": table_col,
        "4": cot_col,
        "5": "all",  # добавить во все коллекции
    }

    while True:
        print(t("choose_collection"))
        
        for key, col_name in [("1", t("mode_1")), ("2", t("mode_2")), ("3", t("mode_3")), ("4", t("mode_4")), ("5", t("all_collections"))]:
            print(f"{col_name}")
        
        col_choice = input(t("your_choice")).strip()
        
        if col_choice not in USER_COLLECTIONS:
            print(t("invalid_choice") + "\n")
            continue
        
        if USER_COLLECTIONS[col_choice] == "all":
            for col in [book_col, reco_col, table_col, cot_col]:
                add_text_to_collection(text=text, collection=col, meta={"source": "user"})
            print(t("text_added_all") + "\n")
        else:
            collection = USER_COLLECTIONS[col_choice]
            add_text_to_collection(text=text, collection=collection, meta={"source": "user"})
            print(t("text_added") + "\n")
        
        break
    
# пересказ и теги
def extract_summary_and_tags(text: str, model=DEFAULT_CHAT_MODEL):
    """
    Возвращает summary через LLM и тематические тэги через YAKE.
    """
    # Пересказ
    try:
        resp = ollama.chat(
            messages=[
                {"role": "system", "content": t("summary_prompt")},
                {"role": "user", "content": text},
            ],
            model=model,
            options={"temperature": 0, "top_p": 1},
        )
        summary = extract_ollama_content(resp).strip()
    except Exception as e:
        print(t("summary_fail"), e)
        summary = ""

    # Извлечение тегов
    try:
        lang = get_current_lang()
        kw_extractor = yake.KeywordExtractor(lan=lang, n=3, top=10)
        tags = [kw for kw, score in kw_extractor.extract_keywords(text)]
    except Exception as e:
        print(t("summary_fail"), e)
        tags = []

    return summary, tags


# ----------------------------
# Режимы работы чат-бота
# ----------------------------
def get_modes():
    return {
        "1": {"name": t("mode_1")},
        "2": {"name": t("mode_2")},
        "3": {"name": t("mode_3")},
        "4": {"name": t("mode_4")},
        "5": {"name": t("mode_5")},
        "6": {"name": t("mode_6")}
    }

def get_domain_to_mode():
    return {
        "rag": "1",
        "recommend": "2",
        "table": "3",
        "cot": "4",
    }

def get_menu_text(skip=False):
    lines = []
    modes = get_modes()
    n = 0
    for key, info in modes.items():
        n = int(key)
        lines.append(info["name"])
    if skip:
        lines.append(f"{n+1} – {t('mode_skip')}")
    return "\n".join(lines)


# ----------------------------
# Чат (CLI)
# ----------------------------
def chat_loop():
    client = get_client()
    book_col = get_or_create_collection(client, BOOKS_COLLECTION)
    reco_col = get_or_create_collection(client, RECO_COLLECTION)
    table_col = get_or_create_collection(client, TABLE_COLLECTION)
    cot_col = get_or_create_collection(client, COT_COLLECTION)

    d(f"default lang: {get_current_lang()}")

    print(t("chat_started") + "\n")
    while True:
        try:
            # выбор режима
            print(t("choose_mode") + get_menu_text(skip=True) + "\n")
            choice = input(t("query_choice")).strip()
                
            # Ввод запроса
            query = input(t("query_prompt")).strip()
            if query.lower() in {"exit", "quit"}:
                break

            # Беседа должна вестить на выбранном пользователем языке
            # Используем ML-классификатор для определения языка запроса
            set_lang(lang_classifier.predict(query))

            d(f"classified lang: {get_current_lang()}")

            # Любой запрос пользователя должен проходить через моральный фильтр
            if not moral_filter.is_allowed(query):
                print(t("moral_block_bot") + "\n")
                continue
            
            modes = get_modes()
            # Если пользователь не выбрал режим вручную
            if choice not in modes:
                predicted_domain = domain_classifier.predict(query)
            
                d(f"predicted domain: {predicted_domain}")
                
                if predicted_domain:
                    domain_to_mode = get_domain_to_mode()
                    choice = domain_to_mode[predicted_domain]
                else:
                    print(t("query_type_undetected") + "\n")
                    print(t("choose_mode") + get_menu_text() + "\n")

                    choice = input(t("query_choice")).strip()

                    if choice not in modes:
                        print(t("invalid_choice") + "\n")
                        continue

            d(f"mode: {choice}")

            if choice == "1":
                hits = retrieve_topk(query, book_col)
            elif choice == "2":
                hits = retrieve_topk(query, reco_col)
            elif choice == "3":
                table = extract_table_from_text(query, collection=table_col)
                if table:
                    print("\n", t("bot_prefix"), "\n", render_markdown_table(table))
                else:
                    print(t("table_fail_bot") + "\n")
                continue
            elif choice == "4":
                solution = solve_complex_task_with_steps(query, cot_col)
                print("\n", t("bot_prefix"), "\n", solution)
                continue
            elif choice == "5":
                add_to_db(text=query, book_col=book_col, reco_col=reco_col, table_col=table_col, cot_col=cot_col)
                continue
            elif choice == "6":
                summary, tags = extract_summary_and_tags(query)
                if summary:
                    print(t("summary_result", summary=summary, tags=", ".join(tags)))
                else:
                    print(t("summary_fail"))
                continue

            prompt = build_prompt(query, hits)
            resp = ollama.chat(
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                model=DEFAULT_CHAT_MODEL,
                options={"temperature": 0, "top_p": 1},
            )
            content = extract_ollama_content(resp)
            print("\n", t("bot_prefix"), content, "\n")
        except KeyboardInterrupt:
            print(t("exit"))
            break
        except Exception as e:
            print(t("error"), e)


# ----------------------------
# Telegram Bot (telebot)
# ОГРАНИЧЕННЫЙ функционал, оставлен как пример интеграции с Telegram
# Для использования всех функций бота используйте CLI `python chatbot.py --chat`
# ----------------------------
USER_STATE = {}  # user_id -> выбранный режим
USER_LANG = {}  # user_id -> lang


def register_telebot_handlers(bot_instance):
    @bot_instance.message_handler(commands=["start"])
    def start(message):
        USER_STATE.pop(message.from_user.id, None)

        # Locale Telegram используется только как fallback до первого текста
        # Основной механизм — ML-классификатор при вводе запроса
        set_lang(message.from_user.language_code)
        USER_LANG[message.from_user.id] = get_current_lang()

        bot_instance.reply_to(
            message,
            t("tg_greeting"),
            reply_markup=main_menu_keyboard(),
        )

    @bot_instance.message_handler(content_types=["text"])
    def text_handler(message):
        tg_process_message(message)


def main_menu_keyboard():
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    modes = get_modes()
    for key, info in modes.items():
        kb.row(KeyboardButton(info["name"]))
    return kb


def tg_process_message(message):
    user_id = message.from_user.id
    query = message.text.strip()

    # Беседа должна вестить на выбранном пользователем языке
    # Используем ML-классификатор для определения языка запроса
    set_lang(lang_classifier.predict(query))
    USER_LANG[user_id] = get_current_lang()

    # Любой запрос пользователя должен проходить через моральный фильтр
    if not moral_filter.is_allowed(query):
        bot.reply_to(
            message,
            t("tg_moral_block"),
        )
        return

    # Если пользователь еще не выбрал режим
    if user_id not in USER_STATE:
        predicted_domain = domain_classifier.predict(query)
        if predicted_domain:
            domain_to_mode = get_domain_to_mode()
            USER_STATE[user_id] = domain_to_mode[predicted_domain]
        else:
            bot.reply_to(message, t("choose_mode") + get_menu_text())
            USER_STATE[user_id] = None
            return

    modes = get_modes()

    # Если режим ещё не выбран — выбираем
    if USER_STATE[user_id] is None:
        if query not in [info["name"] for info in modes.values()]:
            bot.reply_to(message, t("tg_press_button"))
            return
        USER_STATE[user_id] = query
        bot.reply_to(message, t("tg_enter_query"))
        return

    mode = USER_STATE[user_id]

    client = get_client()
    book_col = get_or_create_collection(client, BOOKS_COLLECTION)
    reco_col = get_or_create_collection(client, RECO_COLLECTION)
    cot_col = get_or_create_collection(client, COT_COLLECTION)

    if mode == modes["1"]["name"]:
        hits = retrieve_topk(query, book_col)
        prompt = build_prompt(query, hits)
        resp = ollama.chat(
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            model=DEFAULT_CHAT_MODEL,
            options={"temperature": 0},
        )
        answer = extract_ollama_content(resp)
        bot.reply_to(message, answer)

    elif mode == modes["2"]["name"]:
        hits = retrieve_topk(query, reco_col)
        prompt = build_prompt(query, hits)
        resp = ollama.chat(
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            model=DEFAULT_CHAT_MODEL,
            options={"temperature": 0},
        )
        answer = extract_ollama_content(resp)
        bot.reply_to(message, answer)

    elif mode == modes["3"]["name"]:
        table = extract_table_from_text(query)
        if table:
            bot.reply_to(message, render_table_for_tg(table), parse_mode="Markdown")
        else:
            bot.reply_to(message, t("table_fail_bot"))

    elif mode == modes["4"]["name"]:
        answer = solve_complex_task_with_steps(query, cot_col)
        bot.reply_to(message, answer)

    # После обработки запроса — сбрасываем режим
    USER_STATE[user_id] = None
    bot.reply_to(
        message,
        t("tg_next"),
        reply_markup=main_menu_keyboard(),
    )


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--telegram-token", type=str, help="Telegram bot token")
    args = parser.parse_args()

    if args.build:
        if BOOKS_DIR.exists():
            build_db(BOOKS_DIR, BOOKS_COLLECTION, rebuild=args.rebuild)
        if RECO_DIR.exists():
            build_db(RECO_DIR, RECO_COLLECTION, rebuild=args.rebuild)
        if TABLE_DIR.exists():
            build_db(TABLE_DIR, TABLE_COLLECTION, rebuild=args.rebuild)
        if COT_DIR.exists():
            build_db(COT_DIR, COT_COLLECTION, rebuild=args.rebuild)
    if args.chat or args.telegram:
        global lang_classifier
        lang_classifier = LanguageClassifier()
        global moral_filter
        moral_filter = MoralFilter()
        global domain_classifier
        domain_classifier = DomainClassifier()

        if args.chat:
            # System locale используется только как fallback до первого текста
            # Основной механизм — ML-классификатор при вводе запроса
            chat_loop()

        if args.telegram:
            if not args.telegram_token:
                print("Ошибка: нужно передать токен Telegram через --telegram-token")
                return

            global bot
            bot = telebot.TeleBot(args.telegram_token)

            register_telebot_handlers(bot)

            print(t("tg_started"))
            bot.infinity_polling()
    if not args.build and not args.chat:
        parser.print_help()


if __name__ == "__main__":
    main()
