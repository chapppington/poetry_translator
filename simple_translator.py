#!/usr/bin/env python3
"""
Простой переводчик стихотворений с полными рифмами
"""

import requests
import os
import pickle
import sys

OLLAMA_URL = "http://localhost:11434"
GENERATION_MODEL = "gemma3:12b"
INDEX_DIR = "rag_index"


def load_index(index_name="default"):
    """Загрузить индекс"""
    index_file = os.path.join(INDEX_DIR, f"{index_name}.pkl")
    if not os.path.exists(index_file):
        print(f"Индекс {index_name} не найден")
        return []
    with open(index_file, "rb") as f:
        return pickle.load(f)


def find_poems_by_translator(translator_name, index):
    """Найти все стихотворения переводчика"""
    found = []
    translator_lower = translator_name.lower()
    for chunk in index:
        chunk_text = chunk.get("text", "").lower()
        if translator_lower in chunk_text:
            found.append(chunk)
    return found


def translate(poem_text, translator_style, index, model=GENERATION_MODEL):
    """Перевести стихотворение строго рифмованно, на всю длину"""

    print(f"Поиск стихотворений переводчика {translator_style}...")
    translator_poems = find_poems_by_translator(translator_style, index)
    print(f"Найдено: {len(translator_poems)} стихотворений")

    examples = []
    for chunk in translator_poems:
        chunk_text = chunk.get("text", "")
        if "Оригинальный текст:" in chunk_text and "Перевод:" in chunk_text:
            examples.append(chunk_text)

    if examples:
        examples_text = "\n\n---\n\n".join(examples)
        prompt = f"""Ты профессиональный переводчик детских стихотворений с английского на русский.
⚠️ Задача: перевести это стихотворение **полностью рифмованно** по всей длине.
⚠️ Каждая пара строк должна строго рифмоваться (AABB или ABAB). 
⚠️ Смысл можно слегка адаптировать ради рифмы, но смысл каждой строки должен быть понятен.
⚠️ Сначала **определи все рифмы для всего стихотворения**, а потом составь строки.
⚠️ Весь текст должен быть рифмованным, без исключений, без лишних слов.

Примеры переводов {translator_style} служат как вдохновение, не копируй их дословно:

{examples_text}

---

Переведи это стихотворение строго рифмованно, соблюдая схему рифмы, ритм и музыкальность. Сначала придумай рифмы для всего стихотворения, потом составь строки:

{poem_text}

Твой перевод (только текст, полностью рифмованно, без комментариев):"""
    else:
        prompt = f"""Ты профессиональный переводчик детских стихотворений с английского на русский.
⚠️ Задача: перевести это стихотворение **полностью рифмованно** по всей длине.
⚠️ Каждая пара строк должна строго рифмоваться (AABB или ABAB). 
⚠️ Смысл можно слегка адаптировать ради рифмы, но смысл каждой строки должен быть понятен.
⚠️ Сначала **определи все рифмы для всего стихотворения**, а потом составь строки.
⚠️ Весь текст должен быть рифмованным, без исключений, без лишних слов.

{poem_text}

Твой перевод (только текст, полностью рифмованно, без комментариев):"""

    url = f"{OLLAMA_URL}/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"Ошибка: {e}")
        return None



def main():
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python simple_translator.py <стихотворение> --style <переводчик>")
        print("  python simple_translator.py --file <файл> --style <переводчик>")
        sys.exit(1)

    translator_style = ""
    poem_text = ""
    file_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--style" and i + 1 < len(args):
            translator_style = args[i + 1]
            i += 2
        elif args[i] == "--file" and i + 1 < len(args):
            file_path = args[i + 1]
            i += 2
        else:
            poem_text = " ".join(args[i:])
            break

    if not translator_style:
        print("Ошибка: укажите --style <переводчик>")
        sys.exit(1)

    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            poem_text = f.read().strip()
    elif not poem_text:
        print("Ошибка: укажите стихотворение")
        sys.exit(1)

    # Загружаем индекс
    index = load_index()

    print("\nСтихотворение:")
    print("-" * 50)
    print(poem_text)
    print("-" * 50)
    print(f"\nСтиль: {translator_style}\n")

    # Переводим
    translation = translate(poem_text, translator_style, index)
    if translation:
        print(translation)


if __name__ == "__main__":
    main()
