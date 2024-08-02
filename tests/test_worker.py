import pytest

from rq import Worker, Queue
from rq.job import Job
from fakeredis import FakeStrictRedis, FakeServer

from label_studio_berq.worker import LabelStudioBEWorker
from label_studio_berq.api.models import SetupModel
from rq.worker_registration import get_keys as get_rq_worker_keys


def test_LabelStudioBEWorker_init(fakeredis):

    queue = Queue("test-queue", connection=fakeredis)
    worker = LabelStudioBEWorker([queue], connection=fakeredis)
    worker.bootstrap()

    worker_keys = get_rq_worker_keys(queue, fakeredis)
    assert worker.name == worker_keys.pop().split(":")[-1]


def test_LabelStudioBEWorker_get_project_setup_json(fakeredis):

    project = "3.111"
    model = SetupModel(
        **{
            "project": project,
            "schema": "<View></View>",
            "hostname": "http://localhost:8080",
            "access_token": "82d2dedc2777c40db97f4cc33a81d804886d96f1",
            "extra_params": {"a": "aaa", "b": 2.0},
        }
    )

    fakeredis.hset(
        f"lsberq:project:{project}",
        "setup",
        value=model.model_dump_json(),
    )

    queue = Queue("test-queue", connection=fakeredis)
    job = queue.enqueue("get_project_setup_json", project)
    worker = LabelStudioBEWorker([queue], connection=queue.connection)
    assert worker.work(burst=True)
    job.refresh()
    assert SetupModel.model_validate_json(job.result) == model


def test_execute_job_version(fakeredis):

    queue = Queue("test-queue", connection=fakeredis)
    job = queue.enqueue("get_model_version")
    worker = LabelStudioBEWorker([queue], connection=queue.connection)
    assert worker.work(burst=True)
    job.refresh()
    assert job.result == "0.0.1"
