import os
import pickle
import hashlib
import yaml
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# Шляхи до YAML-файлу з інформацією, директорії для збереження індексу та файлу з кешем хешу
DATA_FILE = "Data/shop_info.yaml"
INDEX_DIR = "Data/faiss_index"
HASH_CACHE_FILE = "Data/shop_info_hash_cache.pkl"

load_dotenv(dotenv_path=".env")

# Отримання ключа API для OpenAI
OPENAI_API_KEY = os.getenv("GPT_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def load_documents_from_yaml(filepath: str):
    """
    Завантажує дані з YAML-файлу та повертає список Document.
    Кожен ключ у YAML-файлі стає окремим документом із заголовком та вмістом.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    documents = []
    for section, content in data.items():
        if isinstance(content, list):
            # Якщо вміст представлено списком (наприклад, для "Цінності")
            content = "\n".join(content)
        combined_content = f"**{section}**\n{content}"
        documents.append(Document(page_content=combined_content, metadata={"section": section}))
    return documents


def compute_file_hash(file_path, algorithm='md5'):
    """
    Обчислює хеш файлу за допомогою вказаного алгоритму.
    """
    hash_algo = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_algo.update(chunk)
    return hash_algo.hexdigest()


# Обчислення поточного хешу YAML-файлу
current_data_hash = compute_file_hash(DATA_FILE)

# Перевірка кешованого хешу
index_up_to_date = False
if os.path.exists(HASH_CACHE_FILE):
    try:
        with open(HASH_CACHE_FILE, 'rb') as f:
            cached_hash = pickle.load(f)
        if cached_hash == current_data_hash:
            index_up_to_date = True
        else:
            print("YAML file changed. Recomputing embeddings...")
    except Exception as e:
        print("Помилка завантаження кешу хешу:", e)

# Якщо кеш актуальний та індекс існує – завантажуємо його, інакше – створюємо новий індекс
if index_up_to_date and os.path.exists(INDEX_DIR):
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    if os.path.exists(DATA_FILE):
        docs = load_documents_from_yaml(DATA_FILE)
    else:
        raise FileNotFoundError(f"Не знайдено файл даних: {DATA_FILE}")
    print("Computing info embeddings...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    # Оновлюємо кеш хешу файлу даних
    with open(HASH_CACHE_FILE, 'wb') as f:
        pickle.dump(current_data_hash, f)


@tool("shop_info_tool")
def shop_info_tool() -> str:
    """
    Інструмент повертає всю інформацію про компанію "Аврора".
    Дані завантажуються із векторного сховища, побудованого з YAML-файлу.
    """
    results = vectorstore.similarity_search("", k=100)
    if results:
        return "\n\n".join([doc.page_content for doc in results])
    else:
        return "Інформація про магазин відсутня."
