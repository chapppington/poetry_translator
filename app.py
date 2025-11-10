#!/usr/bin/env python3
"""
Простое приложение для запросов к Ollama
"""

import requests
import json
import sys


OLLAMA_URL = "http://localhost:11434"


def list_models():
    """Получить список доступных моделей"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return models
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении списка моделей: {e}")
        return []


def generate(prompt, model="gpt-oss:20b", stream=False):
    """Сгенерировать ответ от модели"""
    url = f"{OLLAMA_URL}/api/generate"
    data = {"model": model, "prompt": prompt, "stream": stream}

    try:
        response = requests.post(url, json=data, stream=stream)
        response.raise_for_status()

        if stream:
            # Потоковый режим
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        print(chunk["response"], end="", flush=True)
                    if chunk.get("done", False):
                        print()  # Новая строка в конце
                        break
        else:
            # Обычный режим
            result = response.json()
            return result.get("response", "")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к модели: {e}")
        return None


def chat(messages, model="gpt-oss:20b"):
    """Отправить сообщения в чат"""
    url = f"{OLLAMA_URL}/api/chat"
    data = {"model": model, "messages": messages}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к модели: {e}")
        return None


def main():
    """Главная функция"""
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python app.py <prompt>              # Запрос к модели")
        print("  python app.py --list                   # Список моделей")
        print("  python app.py --model <name> <prompt>  # Указать модель")
        print("  python app.py --stream <prompt>        # Потоковый вывод")
        sys.exit(1)

    if sys.argv[1] == "--list":
        print("Доступные модели:")
        models = list_models()
        for model in models:
            print(f"  - {model.get('name', 'unknown')}")
        return

    # Определяем модель и промпт
    model = "gpt-oss:20b"
    prompt = ""
    stream = False

    args = sys.argv[1:]
    if args[0] == "--model" and len(args) >= 3:
        model = args[1]
        prompt = " ".join(args[2:])
    elif args[0] == "--stream":
        prompt = " ".join(args[1:])
        stream = True
    else:
        prompt = " ".join(args)

    if not prompt:
        print("Ошибка: не указан промпт")
        sys.exit(1)

    print(f"Модель: {model}")
    print(f"Запрос: {prompt}\n")
    print("Ответ:")
    print("-" * 50)

    if stream:
        generate(prompt, model, stream=True)
    else:
        response = generate(prompt, model)
        if response:
            print(response)


if __name__ == "__main__":
    main()
