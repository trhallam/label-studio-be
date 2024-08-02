import asyncio
import time

from label_studio_berq.api.utils import (
    get_rq_worker_info,
    get_rq_available_queues,
    get_rq_worker_status,
    get_model_version,
    get_project_setup,
    get_model_prediction,
)


def test_get_rq_worker_info(redisdb, rqworker, patch_redis):
    with rqworker as _:
        worker_info = get_rq_worker_info(connection=redisdb)
        worker_info = asyncio.run(worker_info)
        assert len(worker_info) == 1
        key, values = worker_info.popitem()
        assert values[b"queues"] == b"pytest1,pytest2"


def test_get_rq_worker_status(redisdb, rqworker, patch_redis):
    with rqworker as _:
        worker_status = get_rq_worker_status()
        worker_status = asyncio.run(worker_status)
    assert worker_status[rqworker.worker.key]["queues"] == b"pytest1,pytest2"
    assert worker_status[rqworker.worker.key]["state"] == b"idle"


def test_get_rq_available_queues(redisdb, rqworker, patch_redis):
    with rqworker as _:
        queue_info = get_rq_available_queues(connection=redisdb)
        result = asyncio.run(queue_info)
    assert len(result) == 2


def test_get_model_version(redisdb, rqworker, rqqueue, patch_redis):
    with rqworker as _:
        queue = rqqueue[0]
        model_version = get_model_version(queue)
        model_version = asyncio.run(model_version)
    assert model_version == "0.1.0"


# def test_get_project_setup(redisdb, rqworker, rqqueue, patch_redis):
#     with rqworker as _:
#         queue = rqqueue[0]
#         setup = get_project_setup(queue, )
