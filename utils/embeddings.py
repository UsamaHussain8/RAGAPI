from langchain_openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions

from config import configs

def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=configs.OPENAI_API_KEY)

def get_embeddings_function():
    return embedding_functions.OpenAIEmbeddingFunction(api_key=configs.OPENAI_API_KEY, model_name=configs.EMBEDDINGS_MODEL)