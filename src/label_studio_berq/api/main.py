from typing import Dict, List, Any

from fastapi import FastAPI, Request, exceptions
from fastapi.responses import JSONResponse
from rq import Queue
import json
import asyncio

from pydantic import ValidationError

from .models import SetupModel, PredictModel
from .utils import (
    get_rq_available_queues,
    get_redis_connection,
    get_model_version,
    get_rq_worker_status,
    get_project_setup,
)

app = FastAPI()

REDIS_PROJECT_PREFIX = "lsberq:project:"


@app.get("/")
async def root():
    return {"message": "label-studio-be API Server"}


@app.get("/{queue_id}/health", description="The health of the API Server")
async def health(queue_id: str):
    queues_json = get_rq_available_queues()
    if queue_id in queues_json:
        return {"status": "UP", "MODEL_CLASS": f"{queue_id}"}
    else:
        return JSONResponse(
            status_code=400,
            content={"msg": f"Error: No model with queue id {queue_id}"},
        )


@app.get("/metrics", description="Metrics for the API Server")
async def metric():
    return {}


@app.post("/{queue_id}/setup")
async def setup(queue_id: str, request: SetupModel):

    connection = get_redis_connection()
    queues_json = get_rq_available_queues()
    queue_workers = queues_json.get(queue_id, None)

    project_key = f"{REDIS_PROJECT_PREFIX}{request.project}"
    connection.hset(project_key, "setup", request.model_dump_json())
    # Deserialize in Worker as
    # SetupModel.model_validate_json(connection.hget(project_key, "setup"))

    if queue_workers:
        queue = Queue(connection=connection, name=queue_id)
        model_version = get_model_version(queue)
        # validate worker can access setup
        redis_setup = get_project_setup(queue, request.project)

        results = asyncio.gather(model_version, redis_setup)

        response = JSONResponse(
            content={"model_version": model_version, "nworkers": len(queue_workers)}
        )
    else:
        response = JSONResponse(
            status_code=400, content={"msg": f"ERROR: Unknown queue id {queue_id}"}
        )
    return response


@app.post("/{queue_id}/predict")
# async def predict(request: PredictModel, queue_id: str):
async def predict(request: Request, queue_id: str):
    # TODO: Get things from request into redis (like extra_params)

    print(await request.body())

    queues_json = get_rq_available_queues()
    queue_workers = queues_json.get(queue_id, None)

    if queue_workers:
        connection = get_redis_connection()
        queue = Queue(connection=connection, name=queue_id)

        model_version = get_model_version(queue)
        response = JSONResponse(
            content={"predict": model_version, "nworkers": len(queue)}
        )
    else:
        response = JSONResponse(
            status_code=400, content={"msg": f"ERROR: Unknown queue id {queue_id}"}
        )
    return response


@app.get("/status")
async def status():
    return get_rq_worker_status()


@app.get("/rq-queues")
async def rq_queues():
    queues_json = get_rq_available_queues()
    return dict(queues_json)


@app.exception_handler(exceptions.RequestValidationError)
@app.exception_handler(ValidationError)
async def _(request: Request, exc: ValidationError):
    print(f"The client sent invalid data!: {exc}")
    exc_json = json.loads(exc.json())
    response = {"message": [], "data": None}
    for error in exc_json:
        response["message"].append(f"{error['loc']}: {error['msg']}")

    return JSONResponse(response, status_code=422)
