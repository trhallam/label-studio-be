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
    get_model_predictions,
)

app = FastAPI()

REDIS_PROJECT_PREFIX = "lsberq:project:"


@app.get("/")
async def root():
    return {"message": "label-studio-be API Server"}


@app.get("/{queue_id}/health", description="The health of the API Server")
async def health(queue_id: str):
    queues_json = await get_rq_available_queues()
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
    """Setup is called by LabelStudio when connecting to a Backend ML model.

    Here, we store the request in redis so that workers can access it - especially
    the extra_params - provided as a json string from LabelStudio - which can be used to control the model.

    For the setup to work, a worker must be active with an appropriate queue_id.
    """
    connection = get_redis_connection()
    queues_json = await get_rq_available_queues()
    queue_workers = queues_json.get(queue_id, None)

    project_key = f"{REDIS_PROJECT_PREFIX}{request.project}"
    connection.hset(project_key, "setup", request.model_dump_json())

    if queue_workers:
        queue = Queue(queue_id, connection=connection)
        model_version, project_setup = await asyncio.gather(
            get_model_version(queue), get_project_setup(queue, request.project)
        )

        response = JSONResponse(
            content={"model_version": model_version, "nworkers": len(queue_workers)}
        )
    else:
        response = JSONResponse(
            status_code=400, content={"msg": f"ERROR: Unknown queue id {queue_id}"}
        )
    return response


@app.post("/{queue_id}/predict")
async def predict(request: PredictModel, queue_id: str):
    # TODO: Get things from request into redis (like extra_params)

    queues_json = await get_rq_available_queues()
    queue_workers = queues_json.get(queue_id, None)

    if queue_workers:
        connection = get_redis_connection()

        project_setup = connection.hget(request.project, "setup")
        print(project_setup)

        queue = Queue(connection=connection, name=queue_id)

        predictions = await get_model_predictions(
            queue, project_setup.setup, request.tasks
        )
        response = JSONResponse(content={})
    else:
        response = JSONResponse(
            status_code=400, content={"msg": f"ERROR: Unknown queue id {queue_id}"}
        )
    return response


@app.get("/status")
async def status():
    return await get_rq_worker_status()


@app.get("/rq-queues")
async def rq_queues():
    queues_json = await get_rq_available_queues()
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
