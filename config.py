from dotenv import dotenv_values
from collections import namedtuple

config = dotenv_values(".env")
Constants = namedtuple('Constants', ['OPENAI_API_KEY', 'EMBEDDINGS_MODEL', 'CHAT_MODEL', 'REDIS_URL', 'MONGO_CONNECTION_STRING', 'MONGODB_COLLECTION_NAME', 'MONGODB_DATABASE_NAME', 'PERSIST_DIRECTORY', 'UPLOADS_FOLDER', 'COLLECTION_NAME'])
configs = Constants(
    OPENAI_API_KEY=config.get("OPENAI_API_KEY"),
    EMBEDDINGS_MODEL=config.get("EMBEDDINGS_MODEL"),
    CHAT_MODEL=config.get("CHAT_MODEL"),
    REDIS_URL=config.get("REDIS_URL"),
    MONGO_CONNECTION_STRING=config.get("MONGO_CONNECTION_STRING"),
    MONGODB_COLLECTION_NAME=config.get("MONGODB_COLLECTION_NAME"),
    MONGODB_DATABASE_NAME=config.get("MONGODB_DATABASE_NAME"),
    PERSIST_DIRECTORY=config.get("PERSIST_DIRECTORY"), 
    UPLOADS_FOLDER=config.get("UPLOADS_FOLDER"),
    COLLECTION_NAME=config.get("COLLECTION_NAME")
)