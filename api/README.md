# OpenAI Chat API Backend

FastAPI backend service for streaming chat using OpenAI's API.

## Prerequisites

- [`uv`](https://github.com/astral-sh/uv) package manager (`pip install uv`)
- OpenAI API key

## Setup

1. Install dependencies (from repository root):
```bash
uv sync
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-your-key-here
```

## Running Locally

From the repository root, run:

```bash
uv run python api/index.py
```

The server will start on `http://localhost:8000`.

For development with auto-reload:
```bash
uv run uvicorn api.index:app --reload
```

## API Endpoints

- **POST** `/api/chat` - Chat endpoint
- **GET** `/api/health` - Health check

View interactive API docs at `http://localhost:8000/docs` when the server is running. 
