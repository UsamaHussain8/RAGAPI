from redis import Redis
from rq import Queue

redis_connection = Redis(host="redis", port=6379, db=0)
tasks_queue = Queue("rag_tasks", connection=redis_connection)