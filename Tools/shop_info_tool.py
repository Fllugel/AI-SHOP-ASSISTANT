import os
import yaml
import hashlib
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

DATA_FILE = "Data/shop_info.yaml"
INDEX_DIR = "Data/faiss_shop_info_index"
hash_yaml_path = os.path.join(INDEX_DIR, "file_hashes.yaml")

load_dotenv(dotenv_path=".env")

def compute_file_hash(file_path, algorithm='md5'):
    hash_algo = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_algo.update(chunk)
    return hash_algo.hexdigest()

# Compute the current hash of the YAML data file.
current_yaml_hash = compute_file_hash(DATA_FILE)

# Load hash data from file if it exists; otherwise, initialize an empty dictionary.
if os.path.exists(hash_yaml_path):
    with open(hash_yaml_path, 'r', encoding='utf-8') as f:
        hash_data = yaml.safe_load(f) or {}
else:
    hash_data = {}

# Use the YAML file name as the key in the hash dictionary.
yaml_key = os.path.basename(DATA_FILE)
stored_hash = hash_data.get(yaml_key)

# Determine if the FAISS index needs to be rebuilt.
rebuild_index = (stored_hash != current_yaml_hash)

# Initialize embeddings.
OPENAI_API_KEY = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("GPT_API_KEY or OPENAI_API_KEY environment variable must be set")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if not rebuild_index and os.path.exists(INDEX_DIR):
    # Load FAISS index from disk.
    try:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        print("Loaded FAISS index from disk.")
    except Exception as e:
        print(f"Error loading index from disk: {e}. Rebuilding index...")
        rebuild_index = True

if rebuild_index or not os.path.exists(INDEX_DIR):
    # Load data from the YAML file.
    with open(DATA_FILE, 'r', encoding='utf-8') as file:
        shop_data = yaml.safe_load(file)

    # Create Document objects for each section.
    documents = []
    for section, content in shop_data.items():
        # If the content is a list (e.g., for "Values"), join the items.
        if isinstance(content, list):
            content_text = "\n".join(content)
        else:
            content_text = str(content)
        doc_content = f"**{section}**\n{content_text}"
        doc = Document(page_content=doc_content, metadata={"section": section})
        documents.append(doc)

    print("Creating FAISS index for shop information...")
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(INDEX_DIR)

        # Create the index directory if it doesn't exist, and update the hash in the YAML file.
        os.makedirs(INDEX_DIR, exist_ok=True)
        hash_data[yaml_key] = current_yaml_hash
        with open(hash_yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(hash_data, f)
        print("FAISS index created and saved.")
    except Exception as e:
        print(f"Error creating FAISS index (API key may be invalid): {e}")
        # Try to load existing index even if it's outdated
        if os.path.exists(INDEX_DIR):
            try:
                vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                print("Loaded existing FAISS index from disk (may be outdated).")
            except Exception as load_error:
                print(f"Failed to load existing index: {load_error}")
                raise ValueError(f"Cannot create or load FAISS index. Please check your API key. Error: {e}")
        else:
            raise ValueError(f"Cannot create FAISS index and no existing index found. Please check your API key. Error: {e}")

def shop_info() -> str:
    """
    Tool returns all information about the company.
    """
    results = vectorstore.similarity_search("", k=100)
    if results:
        return "\n\n".join([doc.page_content for doc in results])
    else:
        return "No shop information available."