import os
from fastapi import FastAPI
import uvicorn
from routers.indexing import index_router
from routers.chat import chat_router
from dotenv import dotenv_values
from collections import namedtuple

app = FastAPI()

app.include_router(index_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")

config = dotenv_values(".env")
Constants = namedtuple('Constants', ['OPEN_API_KEY', 'EMBEDDINGS_MODEL', 'CHAT_MODEL', 'REDIS_URL', 'MONGO_CONNECTION_STRING'])
configs = Constants(config["OPENAI_API_KEY"], config["EMBEDDINGS_MODEL"], config["CHAT_MODEL"], config['REDIS_URL'], config['MONGO_CONNECTION_STRING'])

if __name__== '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
