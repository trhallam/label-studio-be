import pytest

from rq import Worker, Queue
from rq.job import Job

from label_studio_berq.worker import LabelStudioBEWorker
from label_studio_berq.api.models import SetupModel
from rq.worker_registration import get_keys as get_rq_worker_keys


def test_LabelStudioBEWorker_init(test_services):
    redisdb = test_services["redisdb"]

    queue = Queue("test-queue", connection=redisdb)
    worker = LabelStudioBEWorker([queue], connection=redisdb)
    worker.bootstrap()

    worker_keys = get_rq_worker_keys(queue, redisdb)
    assert worker.name == worker_keys.pop().split(":")[-1]


def test_LabelStudioBEWorker_get_project_setup_json(
    test_services, lsproject_setup, lsproject_json
):
    redisdb = test_services["redisdb"]
    model = SetupModel.model_validate_json(lsproject_json)
    queue = Queue("test-queue", connection=redisdb)
    job = queue.enqueue("get_project_setup_json", model.project)
    worker = LabelStudioBEWorker([queue], connection=queue.connection)
    assert worker.work(burst=True)
    job.refresh()
    assert SetupModel.model_validate_json(job.result) == model


def test_execute_job_version(test_services):
    redisdb = test_services["redisdb"]

    queue = Queue("test-queue", connection=redisdb)
    job = queue.enqueue("get_model_version")
    worker = LabelStudioBEWorker([queue], connection=queue.connection)
    assert worker.work(burst=True)
    job.refresh()
    assert job.result == "0.0.1"


def test_predict(test_services, lsproject_setup, lsproject_json):
    # only test for blank result because basic LSBEWorker has no model
    redisdb = test_services["redisdb"]
    model = SetupModel.model_validate_json(lsproject_json)
    queue = Queue("test-queue", connection=redisdb)
    job = queue.enqueue("predict", model.project, [], context=None)
    worker = LabelStudioBEWorker([queue], connection=queue.connection)
    assert worker.work(burst=True)
    job.refresh()
    assert job.result == []
