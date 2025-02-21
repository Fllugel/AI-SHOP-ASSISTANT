import uvicorn
from dotenv import load_dotenv
from langsmith import utils
from api import app
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(dotenv_path=".env")
utils.tracing_is_enabled()

# Дозволені джерела (origins) для CORS
origins = [
    "https://ai-shop-assistant-production.up.railway.app",
    "https://fllugel.github.io/AI-SHOP-ASSISTANT-UI/chat",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # лише ці джерела матимуть доступ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
