import os
import openai
import yaml
import pickle
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

# Set your OpenAI API key.
openai.api_key = os.getenv("GPT_API_KEY")

# Paths to the YAML data and the cache file.
yaml_file_path = 'Data/recommendations.yaml'
cache_file_path = 'Data/recommendations_embeddings_cache.pkl'

def compute_file_hash(file_path, algorithm='md5'):
    """
    Computes a hash of the given file.
    """
    hash_algo = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_algo.update(chunk)
    return hash_algo.hexdigest()

# Initialize the embeddings instance.
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Compute the current hash of the YAML file.
current_yaml_hash = compute_file_hash(yaml_file_path)

# Initialize variables for holiday data.
holiday_data = None
holidays = None
holiday_embeddings = None

# Attempt to load cached holiday data and embeddings.
if os.path.exists(cache_file_path):
    try:
        with open(cache_file_path, 'rb') as f:
            cache_data = pickle.load(f)
        # Check if the YAML file has not changed.
        if cache_data.get('file_hash') == current_yaml_hash:
            holiday_data = cache_data.get('holiday_data')
            holidays = cache_data.get('holidays')
            holiday_embeddings = cache_data.get('holiday_embeddings')
        else:
            print("YAML file changed. Recomputing embeddings...")
    except Exception as e:
        print("Error loading cache:", e)

# If cache is missing or outdated, load the YAML and compute embeddings.
if holiday_data is None or holidays is None or holiday_embeddings is None:
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        holiday_data = yaml.safe_load(file)
    holidays = list(holiday_data.keys())
    print("Computing holiday embeddings...")
    holiday_embeddings = [embeddings.embed_query(holiday) for holiday in holidays]

    # Save the computed embeddings and the hash to cache.
    cache_data = {
        'file_hash': current_yaml_hash,
        'holiday_data': holiday_data,
        'holidays': holidays,
        'holiday_embeddings': holiday_embeddings,
    }
    with open(cache_file_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print("Cache saved.")

@tool("holiday_info_tool")
def holiday_info_tool(key: str) -> str:
    """
    This tool returns a list of gift categories suitable for the holiday that best matches the input key.
    """
    # Compute the embedding for the input key.
    key_embedding = embeddings.embed_query(key)

    # Calculate cosine similarity between the key embedding and precomputed holiday embeddings.
    similarities = cosine_similarity([key_embedding], holiday_embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_match = holidays[best_match_index]

    return f"Information for '{best_match}': {holiday_data[best_match]}"
