import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

#TODO Перевір чи оновляється бд при оновлені файлу

# Шляхи до файлу з інформацією та директорії для збереження індексу
DATA_FILE = "Data/shop_info.txt"
INDEX_DIR = "Data/faiss_index"

load_dotenv(dotenv_path=".env")

# Отримання ключа API для OpenAI
OPENAI_API_KEY = os.getenv("GPT_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def load_documents_from_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    # Розбиваємо текст на абзаци за допомогою подвійного переносу рядка
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    documents = [Document(page_content=paragraph, metadata={}) for paragraph in paragraphs]
    return documents


# Завантаження або створення індексу векторного сховища
if os.path.exists(INDEX_DIR):
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    if os.path.exists(DATA_FILE):
        docs = load_documents_from_file(DATA_FILE)
    else:
        raise FileNotFoundError(f"Не знайдено файл даних: {DATA_FILE}")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)


@tool("avrora_info_tool")
def avrora_info_tool() -> str:
    """
    The tool returns all information about the store and company.
    """
    # Виконуємо пошук за пустим запитом, щоб отримати всі документи.
    results = vectorstore.similarity_search("", k=100)
    if results:
        return "\n\n".join([doc.page_content for doc in results])
    else:
        return "Інформація про магазин відсутня."
