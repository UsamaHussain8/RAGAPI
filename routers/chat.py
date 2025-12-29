from datetime import datetime

from fastapi import APIRouter, status, Query

from langchain_openai import ChatOpenAI
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.retrieval import get_chroma_retriever
from config import configs
from queues.rag_tasks import process_user_query
from client.rq_client import tasks_queue

chat_router = APIRouter()   

@chat_router.post("/chatpdf/", status_code=status.HTTP_200_OK)
async def pdf_chat(query_params: dict):
    user_id: str = query_params.get('user_id')
    query: str = query_params.get('query')
    session_id: str = user_id + "-" + datetime.now().strftime("%d-%m-%Y")
    
    chat_job = tasks_queue.enqueue(process_user_query, args=(query, session_id))
    if chat_job:
        return {"code": 202, "job_id": chat_job.get_id(), "status": "Task added to the queue"}
    else: 
        return {"error": f"An error occurred while generating the response, the job could not be added to the queue"}
    
    # qa_system_prompt = """
    # You are an assistant for question-answering tasks. \
    # Use the following pieces of retrieved context to answer the question. \
    # You can summarize long documents and provide comprehensive answers based on the context. \
    # If you don't know the answer, just say that you don't know. \
    # Keep the answer concise but informative.
    
    # {context}
    # """
    # qa_prompt = ChatPromptTemplate.from_messages([
    #     ("system", qa_system_prompt),
    #     MessagesPlaceholder("chat_history"),
    #     ("human", "{input}"),
    # ])
    
    # retriever = get_chroma_retriever()
    # # Create RAG chain using LCEL
    # # rag_chain = (
    # #     RunnablePassthrough.assign(
    # #         context=lambda x: format_docs(retriever.invoke(x["input"]))
    # #     )
    # #     | qa_prompt
    # #     | llm
    # #     | StrOutputParser()
    # # )
    # # The chain that generates the answer
    # answer_chain = qa_prompt | llm | StrOutputParser()
    
    # rag_chain = (
    #     RunnablePassthrough.assign(
    #         context=lambda x: format_docs(retriever.invoke(x["input"]))
    #     )
    #     | {
    #         "context": lambda x: x["context"], # Pass context forward (optional, but good)
    #         "input": RunnablePassthrough(), # Pass original input forward
    #         "answer": answer_chain, # The actual LLM generated answer
    #     }
    # )
    
    # # Wrap with message history
    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_message_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    #     database_name=configs.MONGODB_DATABASE_NAME, 
    #     collection_name=configs.MONGODB_COLLECTION_NAME
    # )
    
    # try:
    #     # Invoke the chain
    #     result = conversational_rag_chain.invoke(
    #         {"input": query},
    #         config={"configurable": {"session_id": session_id}},
    #     )
    #     final_answer = result.get('answer', 'Error: Answer not found.')
        
    #     return {'answer': final_answer}
    
    # except Exception as err:
    #     print(f"Error during chat: {err}")
    #     return {"error": f"An error occurred while generating the response: {str(err)}"}

chat_router.get("/result", status_code=status.HTTP_200_OK)
async def get_job_result(job_id: str = Query(..., description="Job ID")):
    result = tasks_queue.fetch_job(job_id=job_id)
    if (result.return_value()):
        return {"status": status.HTTP_200_OK, "result": result}
    else:
        return {"status": status.HTTP_202_ACCEPTED, "result": "Processor working in the backend..."}
    

# Format documents helper function
def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


def get_message_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(connection_string=configs.MONGO_CONNECTION_STRING, 
                                     session_id=session_id, 
                                     collection_name=configs.MONGODB_COLLECTION_NAME,
                                     database_name=configs.MONGODB_DATABASE_NAME
    )