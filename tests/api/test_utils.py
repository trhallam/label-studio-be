import pytest
import asyncio

from label_studio_berq.api.utils import (
    get_rq_worker_info,
    get_rq_available_queues,
    get_rq_worker_status,
    get_model_version,
    get_project_setup,
    get_model_prediction,
)


def test_get_rq_worker_info(test_services):
    redisdb = test_services["redisdb"]
    worker_info = get_rq_worker_info(connection=redisdb)
    worker_info = asyncio.run(worker_info)
    assert len(worker_info) == 1
    key, values = worker_info.popitem()
    assert values[b"queues"] == b"pytest1,pytest2"


def test_get_rq_worker_status(test_services):
    rqworker = test_services["workers"][0]
    worker_status = get_rq_worker_status()
    worker_status = asyncio.run(worker_status)
    assert worker_status[rqworker.key]["queues"] == b"pytest1,pytest2"
    assert worker_status[rqworker.key]["state"] == b"idle"


def test_get_rq_available_queues(test_services):
    redisdb = test_services["redisdb"]
    queue_info = get_rq_available_queues(connection=redisdb)
    result = asyncio.run(queue_info)
    assert len(result) == 2


def test_get_model_version(rqqueue):
    queue = rqqueue[0]
    model_version = asyncio.run(get_model_version(queue))
    assert model_version == "0.0.1"


# def test_get_project_setup(redisdb, rqworker, rqqueue, patch_redis):
#     with rqworker as _:
#         queue = rqqueue[0]
#         setup = get_project_setup(queue, )
