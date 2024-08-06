import pytest
import pathlib
from multiprocessing import Process

from rq import Queue, SimpleWorker

TESTS_PATH = pathlib.Path(__file__).parent

from label_studio_berq.worker import LabelStudioBEWorker

from tests.fixtures_resources import *


@pytest.fixture(scope="function")
def patch_redis(redisdb, monkeypatch):

    def force_pytest_redis(*args, **kwargs):
        return redisdb

    monkeypatch.setattr(
        "label_studio_berq.api.utils.get_redis_connection", force_pytest_redis
    )


class WorkerContext:

    def __init__(self, worker: SimpleWorker, startup_pause: float = 0.5):
        self.worker = worker
        self.startup_pause = startup_pause

    def __enter__(self):
        self.proc = Process(target=self.worker.work)
        self.proc.start()
        self.proc.join(self.startup_pause)
        return self

    def __exit__(self, type, value, traceback):
        try:
            self.proc.terminate()
        except ValueError:
            pass


@pytest.fixture()
def rqqueue(redisdb):
    queue1 = Queue(name="pytest1", connection=redisdb)
    queue2 = Queue(name="pytest2", connection=redisdb)
    return [queue1, queue2]


@pytest.fixture()
def rqworker(redisdb, rqqueue, patch_redis):
    worker = LabelStudioBEWorker(rqqueue, connection=redisdb)

    worker.model_version = "0.1.0"

    ctx = WorkerContext(worker)

    return ctx
