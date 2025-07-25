# app.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения из .env файла
load_dotenv()

# Инициализируем клиента OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
currentsapi_key = os.getenv("CURRENTS_API_KEY")

# Проверка, что ключи заданы
if not client.api_key or not currentsapi_key:
    raise ValueError("OPENAI_API_KEY и CURRENTS_API_KEY должны быть установлены в .env файле")

app = FastAPI()

class Topic(BaseModel):
    topic: str

# Функция для подсчёта токенов в тексте
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Асинхронная функция для получения последних новостей
async def get_recent_news(topic: str):
    url = "https://api.currentsapi.services/v1/latest-news"
    params = {"language": "en", "keywords": topic, "apiKey": currentsapi_key}

    async with httpx.AsyncClient() as client_http:
        response = await client_http.get(url, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {response.text}")

    news_data = response.json().get("news", [])
    if not news_data:
        return "Свежих новостей не найдено."

    return "\n".join([article["title"] for article in news_data[:5]])

# Асинхронная функция генерации контента
async def generate_content(topic: str):
    recent_news = await get_recent_news(topic)

    try:
        # Генерация заголовка
        prompt_title = f"Придумайте заголовок для статьи на тему '{topic}', учитывая новости:\n{recent_news}"
        title_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_title}],
            max_tokens=20,
            temperature=0.5,
        )
        title = title_response.choices[0].message.content.strip()

        # Подсчёт токенов в prompt_title и в ответе
        title_tokens_prompt = count_tokens(prompt_title)
        title_tokens_response = count_tokens(title)

        # Генерация мета-описания
        prompt_meta = f"Напишите мета-описание для статьи с заголовком: '{title}'."
        meta_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_meta}],
            max_tokens=30,
            temperature=0.5,
        )
        meta_description = meta_response.choices[0].message.content.strip()

        meta_tokens_prompt = count_tokens(prompt_meta)
        meta_tokens_response = count_tokens(meta_description)

        # Генерация полного текста статьи
        prompt_post = f"""Напишите подробную статью на тему '{topic}', используя последние новости:\n{recent_news}. 
Статья должна быть:
1. Информативной и логичной
2. Содержать не менее 1500 символов
3. Иметь структуру с подзаголовками
4. Включать анализ текущих трендов
5. Включать примеры из актуальных новостей
6. Лёгкой для восприятия
"""
        post_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_post}],
            max_tokens=1500,
            temperature=0.5,
        )
        post_content = post_response.choices[0].message.content.strip()

        post_tokens_prompt = count_tokens(prompt_post)
        post_tokens_response = count_tokens(post_content)

        # Возвращаем результат + статистику по токенам
        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content,
            "tokens": {
                "title_prompt_tokens": title_tokens_prompt,
                "title_response_tokens": title_tokens_response,
                "meta_prompt_tokens": meta_tokens_prompt,
                "meta_response_tokens": meta_tokens_response,
                "post_prompt_tokens": post_tokens_prompt,
                "post_response_tokens": post_tokens_response,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")

@app.post("/generate-post")
async def generate_post_api(topic: Topic):
    return await generate_content(topic.topic)

@app.get("/")
async def root():
    return {"message": "Service is running"}

@app.get("/heartbeat")
async def heartbeat():
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
