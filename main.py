from fastapi import FastAPI, APIRouter, HTTPException, Query, status
import uvicorn
from rq.job import Job

from routers.indexing import index_router
from routers.chat import chat_router
from client.rq_client import redis_connection

app = FastAPI()

app.include_router(index_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")

status_router = APIRouter(prefix="/api/v1", status_code=status.HTTP_200_OK)

@status_router.get("/status/{job_id}")
async def get_status(job_id: str = Query(..., description="Job ID")):
    try:
        job = Job.fetch(job_id, connection=redis_connection)
    except Exception:
        raise HTTPException(status_code=404, detail="Job ID not found or expired.")

    state = job.get_status()

    response = {
        "job_id": job_id,
        "status": state,
        "progress": job.meta.get('progress', 0),
        "message": job.meta.get("status_msg", "Initializing..."), 
    }

    if state == "finished":
        response["message"] = "Task completed successfully."
        response["result"] = job.result
        response["progress"] = 100

    elif state == "failed":
        response["message"] = "Task failed during execution."
        # Extract the error message without the full stack trace
        response["error"] = job.exc_info.split('\n')[-2] if job.exc_info else "Unknown error"
        
    elif state == "started":
        response["message"] = job.meta.get("status_msg", "Worker is processing...")

    elif state == "queued":
        response["message"] = "Waiting for an available worker..."

    return response

if __name__== '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
