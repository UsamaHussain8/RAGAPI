import os
import uuid
import shutil
import datetime

from fastapi import File, UploadFile, APIRouter, Form, HTTPException, status

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

from ..main import configs


index_router = APIRouter()

@index_router.post("/trainpdf/", status_code=status.HTTP_201_CREATED)
async def create_upload_file(user_id: str = Form(...), pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.endswith(".pdf"):
        return {"code": 400, "answer": "Only PDF files are allowed."}
    
    pdf_folder_path = f"Training_Data"
    os.makedirs(pdf_folder_path, exist_ok=True)
    
    # Use a unique temporary filename to avoid collisions, especially if multiple users upload the same file name
    unique_filename = f"{uuid.uuid4()}_{pdf_file.filename}"
    file_path = os.path.join(pdf_folder_path, unique_filename)

    try:
        with open(file_path, "wb") as temp_dest_file:
            temp_dest_file.write(await pdf_file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
        
    docs = read_docs(file_path, user_id)
    vectordb = generate_and_store_embeddings(docs, pdf_file)

    if vectordb is None:
        return {"code": 400, "answer": "Error Occurred during Data Extraction from Pdf."}
        
    shutil.rmtree(pdf_folder_path, ignore_errors=True)

    return {"code": "201", "answer": "PDF Embeddings generated successfully"}

@index_router.post("/deletepdf/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pdf_doc(pdf_file: UploadFile = File(...)):
    embeddings = OpenAIEmbeddings(openai_api_key=configs.OPEN_API_KEY)
    client = chromadb.PersistentClient(path="./trained_db")
    collection = client.get_or_create_collection("PDF_Embeddings", embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=configs.OPEN_API_KEY, model_name=configs.EMBEDDINGS_MODEL))
    vectordb = Chroma(persist_directory="./trained_db", embedding_function=embeddings, collection_name = collection.name)
    
    data_associated_with_ids = vectordb.get(where={"source": pdf_file.filename})
    if data_associated_with_ids["ids"]:
        await vectordb.delete(ids=data_associated_with_ids["ids"])
        print(f"Deleted {len(data_associated_with_ids['ids'])} documents")
    
    return {"code": 200, "answer": "PDF Embeddings deleted successfully"}

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

async def generate_and_store_embeddings(documents, pdf_file):
    client = chromadb.PersistentClient(path="./trained_db")
    collection = await client.get_or_create_collection("PDF_Embeddings",
                                                 embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=configs.OPEN_API_KEY,
                                                                                                                model_name=configs.EMBEDDINGS_MODEL))

    try:
        vectordb = Chroma.from_documents(
                    documents,
                    embedding=OpenAIEmbeddings(openai_api_key=configs.OPEN_API_KEY, model=configs.EMBEDDINGS_MODEL),
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