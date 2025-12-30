from fastapi import FastAPI, APIRouter, HTTPException, Query
import uvicorn
from rq.job import Job

from routers.indexing import index_router
from routers.chat import chat_router
from client.rq_client import redis_connection

app = FastAPI()

app.include_router(index_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")

status_router = APIRouter(prefix="/api/v1")
@status_router.get("/status/{job_id}")
async def get_status(job_id: str = Query(..., description="Job ID")):
    try:
        job = Job.fetch(job_id, connection=redis_connection)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": job.get_status(),
        "result": job.result if job.is_finished else None,
        "progress": job.meta.get('progress', 0) 
    }

if __name__== '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
