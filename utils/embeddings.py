from langchain_openai import OpenAIEmbeddings

from config import configs

def get_embeddings_function():
    return OpenAIEmbeddings(api_key=configs.OPENAI_API_KEY, model=configs.EMBEDDINGS_MODEL)