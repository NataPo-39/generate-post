# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Инициализируем FastAPI приложение
app = FastAPI(
    title="Blog Post Generator API",
    description="Генерирует SEO-оптимизированные блог-посты с учетом актуальных новостей из Currents API",
    version="1.0.0"
)

# Читаем API ключи из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

# Проверяем, что ключи заданы
if not OPENAI_API_KEY:
    raise RuntimeError("Не найден OPENAI_API_KEY в переменных окружения.")
if not CURRENTS_API_KEY:
    raise RuntimeError("Не найден CURRENTS_API_KEY в переменных окружения.")

# Инициализируем OpenAI клиент
client = OpenAI(api_key=OPENAI_API_KEY)

# Pydantic-модель для запроса
class GeneratePostRequest(BaseModel):
    topic: str

# Pydantic-модель для ответа
class GeneratePostResponse(BaseModel):
    title: str
    meta_description: str
    post_content: str
    related_news: list

# Функция для получения свежих новостей из Currents API по теме
def get_related_news(topic: str, max_results: int = 3):
    url = "https://api.currentsapi.services/v1/search"
    params = {
        "apiKey": CURRENTS_API_KEY,
        "keywords": topic,
        "language": "ru",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Берём только первые max_results новостей
        news_list = [
            {"title": item.get("title"), "description": item.get("description")}
            for item in data.get("news", [])[:max_results]
        ]
        return news_list
    except Exception as e:
        # Логируем ошибку и возвращаем пустой список, чтобы не падать
        print(f"Ошибка при запросе Currents API: {e}")
        return []

# Функция для генерации поста с учётом новостей
def generate_post_with_news(topic: str, related_news: list):
    # Формируем текстовый контекст из новостей
    news_context = "\n".join(
        [f"- {n['title']}: {n['description']}" for n in related_news]
    )

    try:
        # Генерируем заголовок
        response_title = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Придумайте привлекательный заголовок для блога на тему: {topic}, учитывая эти новости:\n{news_context}"}
            ],
            max_tokens=50,
            temperature=0.7,
        )
        title = response_title.choices[0].message.content.strip()

        # Генерируем мета-описание
        response_meta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Напишите краткое, но информативное мета-описание для поста с заголовком: {title}, учитывая эти новости:\n{news_context}"}
            ],
            max_tokens=100,
            temperature=0.7,
        )
        meta_description = response_meta.choices[0].message.content.strip()

        # Генерируем основной текст поста
        prompt_post = (
            f"Напишите подробный и увлекательный блог-пост на тему: {topic}, "
            f"учитывая эти новости:\n{news_context}\n"
            "Используйте короткие абзацы, подзаголовки, примеры и ключевые слова для лучшего SEO."
        )
        response_post = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt_post}
            ],
            max_tokens=800,
            temperature=0.7,
        )
        post_content = response_post.choices[0].message.content.strip()

        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content
        }
    except Exception as e:
        # Если произошла ошибка в OpenAI API, поднимаем HTTPException
        raise HTTPException(status_code=500, detail=f"Ошибка генерации поста: {e}")

# Эндпоинт для проверки работоспособности сервиса
@app.get("/health", tags=["Service"])
def health_check():
    """
    Проверка здоровья сервиса
    """
    return {"status": "ok"}

# Основной эндпоинт для генерации поста
@app.post("/generate", response_model=GeneratePostResponse, tags=["Post Generation"])
def generate_post(request: GeneratePostRequest):
    """
    Генерация блог-поста с учётом свежих новостей из Currents API
    """
    topic = request.topic

    # Получаем новости по теме
    related_news = get_related_news(topic)

    # Генерируем пост с учетом этих новостей
    post_data = generate_post_with_news(topic, related_news)

    return GeneratePostResponse(
        title=post_data["title"],
        meta_description=post_data["meta_description"],
        post_content=post_data["post_content"],
        related_news=related_news
    )

# Точка входа для запуска сервера через uvicorn
# Запускаем: python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
