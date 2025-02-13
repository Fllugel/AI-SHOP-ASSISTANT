import sys
import uvicorn
from dotenv import load_dotenv
from langsmith import utils
from api import app
from gradio_interface import launch_gradio_interface


load_dotenv(dotenv_path=".env")

utils.tracing_is_enabled()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    else:
        launch_gradio_interface()
