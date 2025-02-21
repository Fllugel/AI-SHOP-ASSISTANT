import uvicorn
from dotenv import load_dotenv
from langsmith import utils
from api import app
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(dotenv_path=".env")
utils.tracing_is_enabled()

# Дозволені джерела для CORS: лише схема та домен (без шляху)
origins = [
    "https://fllugel.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # дозволені джерела
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
