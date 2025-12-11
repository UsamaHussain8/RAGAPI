import os
import shutil
from datetime import datetime
from dotenv import dotenv_values
from collections import namedtuple
from operator import itemgetter
import uuid

from fastapi import FastAPI, File, UploadFile, APIRouter, Request, Form, HTTPException, status

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import chromadb
from chromadb.utils import embedding_functions

config = dotenv_values(".env")
chat_router = APIRouter()

Constants = namedtuple('Constants', ['OPEN_API_KEY', 'EMBEDDINGS_MODEL', 'CHAT_MODEL', 'REDIS_URL', 'MONGO_CONNECTION_STRING'])
configs = Constants(config["OPENAI_API_KEY"], config["EMBEDDINGS_MODEL"], config["CHAT_MODEL"], config['REDIS_URL'], config['MONGO_CONNECTION_STRING'])

@chat_router.post("/trainpdf/", status_code=status.HTTP_201_CREATED)
async def create_upload_file(user_id: str = Form(...), pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.endswith(".pdf"):
        return {"code": 400, "answer": "Only PDF files are allowed."}
    
    pdf_folder_path = f"Training_Data"
    os.makedirs(pdf_folder_path, exist_ok=True)
    
    file_path = os.path.join(pdf_folder_path, pdf_file.filename)
    with open(file_path, "wb") as temp_dest_file:
        temp_dest_file.write(await pdf_file.read())
        
    docs = read_docs(file_path, user_id)
    vectordb = generate_and_store_embeddings(docs, pdf_file, user_id)

    if vectordb is None:
        return {"code": 400, "answer": "Error Occurred during Data Extraction from Pdf. Please check terminal for more details."}
        
    shutil.rmtree(pdf_folder_path, ignore_errors=True)

    return {"code": "201", "answer": "PDF EMBEDDINGS GENERATED SUCCESSFULLY"}

@chat_router.post("/deletepdf/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pdf_doc(pdf_file: UploadFile = File(...)):
    embeddings = OpenAIEmbeddings(openai_api_key=configs.OPEN_API_KEY)
    client = chromadb.PersistentClient(path="./trained_db")
    collection = client.get_or_create_collection("PDF_Embeddings", embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=config["OPENAI_API_KEY"], model_name=configs.EMBEDDINGS_MODEL))
    vectordb = Chroma(persist_directory="./trained_db", embedding_function=embeddings, collection_name = collection.name)
    
    data_associated_with_ids = vectordb.get(where={"source": pdf_file.filename})
    if data_associated_with_ids["ids"]:
        vectordb.delete(ids=data_associated_with_ids["ids"])
        print(f"Deleted {len(data_associated_with_ids['ids'])} documents")
    
    return {"code": 200, "answer": "PDF EMBEDDINGS DELETED SUCCESSFULLY"}   

@chat_router.post("/chatpdf/", status_code=status.HTTP_200_OK)
async def pdf_chat(query_params: dict):
    user_id: str = query_params.get('user_id')
    query: str = query_params.get('query')
    session_id: str = user_id + "-" + datetime.now().strftime("%d/%m/%Y")

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=configs.OPEN_API_KEY)
    client = chromadb.PersistentClient(path="./trained_db")
    collection = client.get_or_create_collection("PDF_Embeddings", 
                                                 embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=config["OPENAI_API_KEY"],
                                                                                                                model_name=configs.EMBEDDINGS_MODEL))
    vectordb = Chroma(persist_directory="./trained_db", 
                      embedding_function=embeddings, 
                      collection_name = collection.name)
    
    """Retrieve the documents relevant to the query and generate the response."""
    # Create retriever
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6}
    )

    relevant_docs = retriever.invoke(query)

    model = configs.CHAT_MODEL
    llm = ChatOpenAI(openai_api_key=configs.OPEN_API_KEY, model = model, temperature = 0.0)
    
    # Create contextualize question prompt for history-aware retrieval
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_system_prompt = """
    You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    You can summarize long documents and provide comprehensive answers based on the context. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer concise but informative.
    
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create RAG chain using LCEL
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    # Wrap with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    try:
        # Invoke the chain
        result = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        
        return {'answer': result}
    
    except Exception as err:
        print(f"Error during chat: {err}")
        return {"error": f"An error occurred while generating the response: {str(err)}"}

# Format documents helper function
def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
def read_docs(pdf_file: str, user_id: str):
    pdf_loader = PyPDFLoader(pdf_file)
    pdf_documents = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(pdf_documents)
    
    now = datetime.now()
    for doc in documents:
        doc.metadata = {
            "user": user_id,
            "id": str(uuid.uuid4()),  
            "source": pdf_file.split("\\")[-1],
            'created_at': now.strftime("%d/%m/%Y %H:%M:%S")
        }

    return documents

def generate_and_store_embeddings(documents, pdf_file, user_id):
    client = chromadb.PersistentClient(path="./trained_db")
    collection = client.get_or_create_collection("PDF_Embeddings",
                                                 embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=config["OPENAI_API_KEY"],
                                                                                                                model_name=configs.EMBEDDINGS_MODEL))

    try:
        vectordb = Chroma.from_documents(
                    documents,
                    embedding=OpenAIEmbeddings(openai_api_key=config["OPENAI_API_KEY"], model=configs.EMBEDDINGS_MODEL),
                    persist_directory='./trained_db',
                    collection_name = collection.name, 
                    client = client
        )
        print(collection.count())
        data_associated_with_ids = vectordb.get(where={"source": pdf_file.filename})
        print(data_associated_with_ids["ids"])

    except Exception as err:
        print(f"An error occured: {err=}, {type(err)=}")
        return {"answer": "An error occured while generating embeddings. Please check terminal for more details."}
    return vectordb

def get_message_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(connection_string=configs.MONGO_CONNECTION_STRING, 
                                     session_id=session_id, 
                                     collection_name="Chat_History")

def retrieve_message_history(runnable, retriever):
    try:
        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_message_history,
            input_messages_key="query",
            history_messages_key="history",
        )
    except Exception as err:
        print(f"Unexpected error occured. {err=}, {type(err)=}")
        return None

    return with_message_history