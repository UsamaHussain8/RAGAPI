import os
import shutil
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from utils.retrieval import get_chroma_retriever
from config import configs
from routers.indexing import read_docs, generate_and_store_embeddings

load_dotenv()

def process_pdf_embeddings_task(file_path, user_id, original_filename):
    docs = read_docs(file_path, user_id, original_filename)
    vectordb = generate_and_store_embeddings(docs, original_filename)

    if vectordb is None:
        return {"code": 400, "answer": "Error Occurred during Data Extraction from PDF."}

    if os.path.exists(file_path):
        os.remove(file_path)

    # shutil.rmtree(file_path, ignore_errors=True)

    return {"code": "201", "answer": "PDF Embeddings generated successfully"}

def process_user_query(query: str, session_id: str):
    model = configs.CHAT_MODEL
    llm = ChatOpenAI(openai_api_key=configs.OPENAI_API_KEY, model = model, temperature = 0.0)

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
    
    chroma_retriever = get_chroma_retriever()

    answer_chain = qa_prompt | llm | StrOutputParser()

    rag_lcel_chain = (
        RunnableParallel(
            context=chroma_retriever, 
            question=RunnablePassthrough()
        )
    | answer_chain
    )

    # Wrap with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_lcel_chain,
        get_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        database_name=configs.MONGODB_DATABASE_NAME, 
        collection_name=configs.MONGODB_COLLECTION_NAME
    )

    try:
        # Invoke the chain
        result = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        final_answer = result.get('answer', 'Error: Answer not found.')
    
        return {'answer': final_answer}

    except Exception as err:
        print(f"Error during chat: {err}")
        return {"error": f"An error occurred while generating the response: {str(err)}"}
    
# Format documents helper function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_message_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(connection_string=configs.MONGO_CONNECTION_STRING, 
                                     session_id=session_id, 
                                     collection_name=configs.MONGODB_COLLECTION_NAME,
                                     database_name=configs.MONGODB_DATABASE_NAME
                                    )