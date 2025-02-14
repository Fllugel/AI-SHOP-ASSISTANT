import os
import yaml
import hashlib
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Шляхи до YAML-файлу з інформацією, директорії для збереження індексу
# та файлу для збереження хешів у форматі YAML.
DATA_FILE = "Data/shop_info.yaml"
INDEX_DIR = "Data/faiss_index"
hash_yaml_path = os.path.join(INDEX_DIR, "file_hashes.yaml")

load_dotenv(dotenv_path=".env")

def compute_file_hash(file_path, algorithm='md5'):
    """Обчислює хеш файлу для визначення змін."""
    hash_algo = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_algo.update(chunk)
    return hash_algo.hexdigest()

# Обчислюємо поточний хеш YAML-файлу з даними.
current_yaml_hash = compute_file_hash(DATA_FILE)

# Завантажуємо дані з файлу хешів, якщо він існує; інакше – ініціалізуємо порожній словник.
if os.path.exists(hash_yaml_path):
    with open(hash_yaml_path, 'r', encoding='utf-8') as f:
        hash_data = yaml.safe_load(f) or {}
else:
    hash_data = {}

# Використовуємо ім'я YAML-файлу як ключ у словнику з хешами.
yaml_key = os.path.basename(DATA_FILE)
stored_hash = hash_data.get(yaml_key)

# Визначаємо, чи потрібно перебудовувати FAISS-індекс.
rebuild_index = (stored_hash != current_yaml_hash)

# Ініціалізуємо embeddings.
OPENAI_API_KEY = os.getenv("GPT_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if not rebuild_index and os.path.exists(INDEX_DIR):
    # Завантажуємо FAISS-індекс з диску.
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Loaded FAISS index from disk.")
else:
    # Завантажуємо дані з YAML-файлу.
    with open(DATA_FILE, 'r', encoding='utf-8') as file:
        shop_data = yaml.safe_load(file)

    # Створюємо об'єкти Document для кожного розділу.
    documents = []
    for section, content in shop_data.items():
        # Якщо вміст представлено списком (наприклад, для "Цінності"), об'єднуємо елементи.
        if isinstance(content, list):
            content_text = "\n".join(content)
        else:
            content_text = str(content)
        doc_content = f"**{section}**\n{content_text}"
        doc = Document(page_content=doc_content, metadata={"section": section})
        documents.append(doc)

    print("Creating FAISS index for shop information...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_DIR)

    # Створюємо директорію для індексу, якщо її немає, та оновлюємо хеш у YAML-файлі.
    os.makedirs(INDEX_DIR, exist_ok=True)
    hash_data[yaml_key] = current_yaml_hash
    with open(hash_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(hash_data, f)
    print("FAISS index created and saved.")

@tool("shop_info_tool")
def shop_info_tool() -> str:
    """
    Інструмент повертає всю інформацію про компанію "Аврора".
    Дані завантажуються із FAISS-індексу, побудованого з YAML-файлу.
    """
    results = vectorstore.similarity_search("", k=100)
    if results:
        return "\n\n".join([doc.page_content for doc in results])
    else:
        return "Інформація про магазин відсутня."
