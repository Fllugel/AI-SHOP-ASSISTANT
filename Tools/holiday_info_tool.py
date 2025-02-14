import os
import openai
import yaml
import hashlib
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# Set your OpenAI API key.
openai.api_key = os.getenv("GPT_API_KEY")

# Paths to the YAML data, FAISS index directory, and the YAML file that stores hashes.
yaml_file_path = 'Data/recommendations.yaml'
INDEX_DIR = 'Data/faiss_holidays_index'
hash_yaml_path = os.path.join(INDEX_DIR, "file_hashes.yaml")


def compute_file_hash(file_path, algorithm='md5'):
    """Compute a hash of a file to detect changes."""
    hash_algo = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_algo.update(chunk)
    return hash_algo.hexdigest()


# Compute the current hash for our recommendations YAML file.
current_yaml_hash = compute_file_hash(yaml_file_path)

# Load hash data from the YAML file if it exists; otherwise, initialize an empty dictionary.
if os.path.exists(hash_yaml_path):
    with open(hash_yaml_path, 'r', encoding='utf-8') as f:
        hash_data = yaml.safe_load(f) or {}
else:
    hash_data = {}

# Use the base name of the YAML file as the key in the hash dictionary.
yaml_key = os.path.basename(yaml_file_path)
stored_hash = hash_data.get(yaml_key)

# Determine whether to rebuild the FAISS index.
rebuild_index = (stored_hash != current_yaml_hash)

# Initialize the embeddings instance.
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

if not rebuild_index:
    # Load the FAISS index from disk.
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Loaded FAISS index from disk.")
else:
    # Load holiday recommendation data from the YAML file.
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        holiday_data = yaml.safe_load(file)

    # Create Document objects for each holiday.
    documents = []
    for holiday, recommendation in holiday_data.items():
        # If recommendations are provided as a list, join them into a single string.
        if isinstance(recommendation, list):
            recommendation_text = "\n".join(recommendation)
        else:
            recommendation_text = str(recommendation)
        doc = Document(page_content=recommendation_text, metadata={"holiday": holiday})
        documents.append(doc)

    print("Creating FAISS index for holiday recommendations...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_DIR)

    # Ensure the index directory exists and update the hash in the YAML hash file.
    os.makedirs(INDEX_DIR, exist_ok=True)
    hash_data[yaml_key] = current_yaml_hash
    with open(hash_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(hash_data, f)
    print("FAISS index created and saved.")


@tool("holiday_info_tool")
def holiday_info_tool(key: str) -> str:
    """
    This tool returns a list of gift categories suitable for the holiday
    that best matches the input key.
    """
    # Query the FAISS index for the most similar document.
    results = vectorstore.similarity_search(key, k=1)
    if results:
        best_match = results[0]
        holiday_name = best_match.metadata.get("holiday", "Unknown Holiday")
        recommendations = best_match.page_content
        return f"Information for '{holiday_name}': {recommendations}"
    else:
        return "No matching holiday information found."
