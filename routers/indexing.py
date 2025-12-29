import os
import uuid
import shutil
import datetime
from typing import List

from fastapi import File, UploadFile, APIRouter, Form, HTTPException, status

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import configs
from utils.embeddings import get_embeddings_function
from utils.retrieval import get_chroma_instance
from queues.rag_tasks import process_pdf_embeddings_task
from client.rq_client import tasks_queue

index_router = APIRouter()

@index_router.post("/uploadpdf/", status_code=status.HTTP_201_CREATED)
async def create_upload_file(user_id: str = Form(...), pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.endswith(".pdf"):
        return {"code": 400, "answer": "Only PDF files are allowed."}
    
    pdf_folder_path = configs.UPLOADS_FOLDER
    os.makedirs(pdf_folder_path, exist_ok=True)
    
    # Use a unique temporary filename to avoid collisions, especially if multiple users upload the same file name
    unique_filename = f"{uuid.uuid4()}_{pdf_file.filename}"
    file_path = os.path.join(pdf_folder_path, unique_filename)

    try:
        with open(file_path, "wb") as temp_dest_file:
            temp_dest_file.write(await pdf_file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    job = tasks_queue.enqueue(
        process_pdf_embeddings_task,
        args=(file_path, user_id, pdf_file.filename)
    )

    return {
        "code": "202", 
        "answer": "Upload received. Processing in background.",
        "job_id": job.get_id() # Give the user a ticket number
    }

@index_router.post("/deletepdf/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pdf_doc(pdf_file: UploadFile = File(...)):
    vectordb = get_chroma_instance()
    
    data_associated_with_ids = vectordb.get(where={"source": pdf_file.filename})
    if data_associated_with_ids["ids"]:
        await vectordb.delete(ids=data_associated_with_ids["ids"])
        print(f"Deleted {len(data_associated_with_ids['ids'])} documents")
    
    return {"code": 200, "answer": "PDF Embeddings deleted successfully"}

def read_docs(pdf_file: str, user_id: str, original_filename: str):
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

def generate_and_store_embeddings(documents: List[Document], pdf_file_name: str):
    embeddings_function = get_embeddings_function()
    try:
        vectordb = Chroma.from_documents(
                    documents,
                    embedding=embeddings_function,
                    persist_directory=configs.PERSIST_DIRECTORY,
                    collection_name = configs.COLLECTION_NAME, 
        )
        vectordb._client.persist()
        print(vectordb._collection.count())
        data_associated_with_ids = vectordb.get(
            where={"source": pdf_file_name},
            include=['ids']       
        )
        print(f"IDs associated with {pdf_file_name}: {data_associated_with_ids['ids']}")

    except Exception as err:
        print(f"An error occured: {err=}, {type(err)=}")
        return {"answer": "An error occured while generating embeddings."}
    return vectordb