from dotenv import load_dotenv
from langsmith import utils
from gradio_interface import launch_gradio_interface

load_dotenv(dotenv_path=".env")
utils.tracing_is_enabled()

if __name__ == "__main__":
    launch_gradio_interface()
