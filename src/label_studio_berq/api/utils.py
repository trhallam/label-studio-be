from typing import Dict, List

import os
from collections import defaultdict
import time
from redis import Redis
from rq.worker_registration import get_keys as get_rq_worker_keys
from rq.utils import decode_redis_hash, as_text
from rq.job import Job
from rq import Queue

LSBE_RQ_REDIS_HOST = os.environ.get("LSBE_RQ_REDIS_HOST", "localhost")
LSBE_RQ_REDIS_PORT = os.environ.get("LSBE_RQ_REDIS_PORT", 6379)


def get_redis_connection(connection: Redis | None = None) -> Redis:
    """Get the Redis server connection using env vars if None provided."""
    if connection is None:
        connection = Redis(
            host=LSBE_RQ_REDIS_HOST,
            port=LSBE_RQ_REDIS_PORT,
        )
    return connection


async def get_rq_worker_info(connection: Redis | None = None) -> Dict:
    """Get information on available rq workers from Redis

    Args:
        connection: Uses the env vars if connection is None.

    Returns:
        RQ worker information from Redis
    """
    connection = get_redis_connection(connection)
    worker_keys = get_rq_worker_keys(connection=connection)
    worker_info = {key: connection.hgetall(key) for key in worker_keys}
    return worker_info


async def get_rq_worker_status():
    binfo = await get_rq_worker_info()
    info_json = {
        as_text(key): decode_redis_hash(values) for key, values in binfo.items()
    }
    return info_json


async def get_rq_available_queues(connection: Redis | None = None):
    """Get available (registered worker) queues and their workers."""
    connection = get_redis_connection(connection)
    worker_keys = get_rq_worker_keys(connection=connection)
    queues = defaultdict(list)
    for worker in worker_keys:
        worker_queues = connection.hget(worker, "queues").decode().split(",")
        for q in worker_queues:
            queues[q] += [as_text(worker)]
    return queues


def get_job_result(job: Job, queue: Queue, refresh_interval: float = 0.1) -> Job:
    """Blocks while waiting for a function to finish.

    Args:
        job: Job to watch
        queue: not needed
        refresh_interval: the time between checks (s)

    Returns:
        finished job
    """
    while True:
        status = job.get_status(refresh=True)
        if status in ["failed", "finished", "cancelled"]:
            break
        time.sleep(refresh_interval)
    return job


async def get_model_version(
    queue: Queue, model_version_func: str = "get_model_version"
) -> str:
    """Ask the worker to return the model version or worker.model_version"""
    job = queue.enqueue(model_version_func)
    job = get_job_result(job, queue)

    if job.result:
        model_version = job.result
    else:
        model_version = ""

    return model_version


async def get_project_setup(queue: Queue, project: str) -> Dict:
    """Ask the worker to recover the project setup json str"""
    job = queue.enqueue("get_project_setup_json", project)
    job = get_job_result(job, queue)

    if job.result:
        return job.result
    else:
        return ""


async def get_model_predictions(
    queue: Queue, tasks: List, context=None, model_prediction_func: str = "predict"
) -> Dict:
    """Passes the predict API call to the backend workers and waits for a
    result.

    Args:
        queue: The rq.Queue object to pass the job to
        tasks: The list of tasks from LabelStudio
        context: The context parameter for interactive predictions from LabelStudio
        model_prediction_func: The name of the Worker class function to use for
            prediction
    Returns:
        results dictionary
    """
    job = queue.enqueue(model_prediction_func, tasks, context=context)
    job = get_job_result(job, queue)

    if job.result:
        return job.result
    else:
        return dict()
