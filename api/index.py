from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from urllib import request as urllib_request, error

# Load .env locally if available; Vercel uses project env vars
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    developer_message: str
    user_message: str
    model: str = "gpt-4o-mini"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY. Set it in your environment or .env file.")
    
    payload = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": request.developer_message},
            {"role": "user", "content": request.user_message}
        ],
    }

    req = urllib_request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {"response": data["choices"][0]["message"]["content"]}
    except error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e.code} {e.reason}\n{body}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")

@app.get("/")
async def root():
    return Response(status_code=204)

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# Vercel serverless function handler
from mangum import Mangum
handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
