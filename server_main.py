import uvicorn
from dotenv import load_dotenv
from langsmith import utils
from api import app

load_dotenv(dotenv_path=".env")
utils.tracing_is_enabled()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
