#!/usr/bin/env python3
"""
RAG: Индексация документов для поиска
"""

import requests
import os
import pickle
import sys
import json
from pathlib import Path
from typing import List, Dict


OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # Модель для embeddings
INDEX_DIR = "rag_index"
CHUNK_SIZE = 500  # Размер чанка в символах
CHUNK_OVERLAP = 50  # Перекрытие между чанками


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Получить embedding для текста"""
    url = f"{OLLAMA_URL}/api/embeddings"
    data = {"model": model, "prompt": text}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("embedding", [])
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении embedding: {e}")
        return []


def split_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Разбить текст на чанки"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Пытаемся разбить по предложениям
        if end < len(text):
            # Ищем последнюю точку, восклицательный или вопросительный знак
            last_sentence = max(
                chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"), chunk.rfind("\n")
            )
            if last_sentence > chunk_size * 0.5:  # Если нашли в середине чанка
                chunk = chunk[: last_sentence + 1]
                end = start + last_sentence + 1

        chunks.append(chunk.strip())
        start = end - overlap  # Перекрытие

        if start >= len(text):
            break

    return chunks


def process_json_file(file_path: str) -> List[str]:
    """Обработать JSON файл и создать текстовые чанки"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Ошибка при чтении JSON файла {file_path}: {e}")
        return []

    chunks = []

    # Создаем структурированное текстовое представление
    text_parts = []

    if "title_original" in data:
        text_parts.append(f"Название (оригинал): {data['title_original']}")
    if "title_translation" in data:
        text_parts.append(f"Название (перевод): {data['title_translation']}")
    if "author" in data:
        text_parts.append(f"Автор: {data['author']}")
    if "translator" in data:
        text_parts.append(f"Переводчик: {data['translator']}")

    if "original" in data:
        text_parts.append(f"\nОригинальный текст:\n{data['original']}")
    if "translation" in data:
        text_parts.append(f"\nПеревод:\n{data['translation']}")

    if "rhyme_scheme" in data:
        text_parts.append(f"\nСхема рифмовки: {data['rhyme_scheme']}")
    if "meter" in data:
        text_parts.append(f"Размер: {data['meter']}")
    if "devices" in data:
        devices_str = (
            ", ".join(data["devices"])
            if isinstance(data["devices"], list)
            else str(data["devices"])
        )
        text_parts.append(f"Художественные приемы: {devices_str}")
    if "comment" in data:
        text_parts.append(f"\nКомментарий: {data['comment']}")

    # Создаем один большой текст для индексации
    full_text = "\n".join(text_parts)

    # Разбиваем на чанки
    chunks = split_text(full_text)

    return chunks


def index_file(file_path: str, metadata: Dict = None) -> List[Dict]:
    """Проиндексировать файл"""
    print(f"Индексация файла: {file_path}")

    file_ext = Path(file_path).suffix.lower()

    # Обработка JSON файлов
    if file_ext == ".json":
        chunks = process_json_file(file_path)
    else:
        # Обычные текстовые файлы
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")
            return []

        chunks = split_text(text)

    indexed_chunks = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        print(f"  Обработка чанка {i + 1}/{len(chunks)}...", end="\r")

        embedding = get_embedding(chunk)
        if not embedding:
            print(
                f"\n  Предупреждение: не удалось получить embedding для чанка {i + 1}"
            )
            continue

        indexed_chunks.append(
            {
                "text": chunk,
                "embedding": embedding,
                "metadata": {"file": file_path, "chunk_index": i, **(metadata or {})},
            }
        )

    print(f"\n  Проиндексировано {len(indexed_chunks)} чанков")
    return indexed_chunks


def index_directory(directory: str, extensions: List[str] = None) -> List[Dict]:
    """Проиндексировать все файлы в директории"""
    if extensions is None:
        extensions = [".txt", ".md", ".py", ".js", ".html", ".css", ".json"]

    all_chunks = []
    path = Path(directory)

    if not path.exists():
        print(f"Директория {directory} не существует")
        return []

    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix in extensions:
            chunks = index_file(str(file_path))
            all_chunks.extend(chunks)

    return all_chunks


def save_index(chunks: List[Dict], index_name: str = "default"):
    """Сохранить индекс"""
    os.makedirs(INDEX_DIR, exist_ok=True)

    index_file = os.path.join(INDEX_DIR, f"{index_name}.pkl")

    with open(index_file, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nИндекс сохранен: {index_file}")
    print(f"Всего чанков: {len(chunks)}")


def main():
    """Главная функция"""
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python rag_index.py <файл>              # Индексировать файл")
        print("  python rag_index.py --dir <директория>  # Индексировать директорию")
        print("  python rag_index.py --name <имя> <файл>  # Указать имя индекса")
        sys.exit(1)

    chunks = []
    index_name = "default"

    args = sys.argv[1:]

    if args[0] == "--dir" and len(args) >= 2:
        directory = args[1]
        if len(args) >= 4 and args[2] == "--name":
            index_name = args[3]
        chunks = index_directory(directory)
    elif args[0] == "--name" and len(args) >= 3:
        index_name = args[1]
        file_path = args[2]
        chunks = index_file(file_path)
    else:
        file_path = args[0]
        chunks = index_file(file_path)

    if chunks:
        save_index(chunks, index_name)
    else:
        print("Не удалось проиндексировать документы")


if __name__ == "__main__":
    main()
