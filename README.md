# Embedding Service

HTTP-сервис для генерации векторных эмбеддингов текстов на основе моделей [SentenceTransformers](https://www.sbert.net/).

## Возможности

- Генерация эмбеддинга для одного текста
- Пакетная генерация эмбеддингов (batch)
- Получение размерности вектора текущей модели
- Асинхронный FastAPI-сервер
- Деплой через Docker / Docker Compose

## Стек

- **Python** 3.11+
- **FastAPI** — HTTP-фреймворк
- **SentenceTransformers** — загрузка и инференс модели
- **Pydantic Settings** — конфигурация через переменные окружения
- **Uvicorn** — ASGI-сервер
- **uv** — управление зависимостями

## Конфигурация

Скопируйте `.env.example` в `.env` и при необходимости измените значения:

```bash
cp .env.example .env
```

| Переменная | По умолчанию | Описание |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `ai-forever/ru-en-RoSBERTa` | Имя модели на HuggingFace Hub |
| `HF_TOKEN` | — | Токен HuggingFace для доступа к приватным моделям |

## Запуск

### Локально

```bash
uv sync
uv run python main.py
```

### Docker Compose

```bash
docker compose up -d
```

Сервис будет доступен на `http://localhost:8001`.

## API

### `POST /embed`

Получить эмбеддинг одного текста.

**Тело запроса:**
```json
{ "text": "Пример текста" }
```

**Ответ:**
```json
{ "vector": [0.123, -0.456, ...] }
```

---

### `POST /embed/batch`

Пакетная генерация эмбеддингов.

**Тело запроса:**
```json
{ "texts": ["Первый текст", "Второй текст"] }
```

**Ответ:**
```json
{ "vectors": [[0.123, ...], [0.456, ...]] }
```

---

### `GET /config/dim`

Получить размерность вектора загруженной модели.

**Ответ:**
```json
{ "dim": 1024 }
```

## Swagger UI

После запуска документация доступна по адресу: `http://localhost:8001/docs`
