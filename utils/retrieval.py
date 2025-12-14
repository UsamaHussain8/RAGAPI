import chromadb
from .embeddings import get_embeddings_function
from langchain_chroma import Chroma

from config import configs

def get_chroma_client():
    return chromadb.PersistentClient(path=configs.PERSIST_DIRECTORY)

def get_chroma_collection():
    client = get_chroma_client()
    embeddings_function = get_embeddings_function()
    collection = client.get_or_create_collection("PDF_Embeddings", embedding_function=embeddings_function)

    return collection

def get_chroma_instance():
    embeddings_function = get_embeddings_function()
    chroma_collection = get_chroma_collection()

    return Chroma(persist_directory=configs.PERSIST_DIRECTORY, 
                      embedding_function=embeddings_function, 
                      collection_name = chroma_collection.name)

def get_chroma_retriever():
    chroma_retriever = get_chroma_instance()
    """Retrieve the documents relevant to the query and generate the response."""
    # Create retriever
    retriever = chroma_retriever.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6}
    )
    
    return retriever