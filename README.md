# generate-post

1.

# Blog Post Generator (FastAPI)

📌 **Что это:**  
Веб‑приложение на FastAPI для генерации статей в блог с учетом последних новостей из Currents API и генерации текста с помощью OpenAI.

## 🚀 Как запустить локально

1. Клонируй репозиторий:
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo


2. Установи зависимости:

pip install -r requirements.txt


3. Создай файл .env:

cp .env.example .env


4. Запусти приложение:

python app.py

Сервис будет доступен по адресу:

http://127.0.0.1:8000



📂 Структура проекта

blog_generator/
├── app.py
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md


✅ Эндпоинты

POST /generate-post — генерация статьи (тело запроса: {"topic": "ваша тема"})

GET / — проверка работы сервиса

GET /heartbeat — статус сервиса


⚠️ Важно: файл .env с ключами никогда не коммить в GitHub!




