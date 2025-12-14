from .embeddings import get_embeddings_function
from langchain_chroma import Chroma

from config import configs

# def get_chroma_client():
#     return chromadb.PersistentClient(path=configs.PERSIST_DIRECTORY)

# def get_chroma_collection():
#     client = get_chroma_client()
#     embeddings_function = get_embeddings_function()
#     collection = client.get_or_create_collection("PDF_Embeddings", embedding_function=embeddings_function)

#     return collection

def get_chroma_instance():
    """
    Initializes and returns the LangChain Chroma vector store instance.
    The Chroma object handles connecting to the persistent directory and 
    creating the collection if it doesn't exist.
    """
    embeddings_function = get_embeddings_function()

    return Chroma(
        persist_directory=configs.PERSIST_DIRECTORY, 
        embedding_function=embeddings_function, 
        collection_name=configs.COLLECTION_NAME
    )

def get_chroma_retriever():
    """
    Creates and returns the Chroma retriever configured for Maximum Marginal Relevance (MMR) search.
    """
    chroma_retriever = get_chroma_instance()
    
    # Create retriever
    retriever = chroma_retriever.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6}
    )
    
    return retriever