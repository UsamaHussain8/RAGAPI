from redis import Redis
from rq import Queue

redis_connection = Redis(host="localhost", port=6379)       # localhost or redis?
tasks_queue = Queue("rag_tasks", connection=redis_connection)