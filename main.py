from fastapi import FastAPI
import uvicorn

from routers.indexing import index_router
from routers.chat import chat_router

app = FastAPI()

app.include_router(index_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")

if __name__== '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
