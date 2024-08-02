import pytest
import json
import yaml
import pathlib
from multiprocessing import Process

from rq.job import Job
from rq import Queue, SimpleWorker

from fakeredis import FakeStrictRedis, FakeServer


import label_studio_sdk._extensions.label_studio_tools.core.utils.io

TESTS_PATH = pathlib.Path(__file__).parent

from label_studio_berq.worker import LabelStudioBEWorker

from tests.fixtures_resources import *


# @pytest.fixture(scope="session")
# def model_sam_vit_h():
#     return get_model("vit_h", TESTS_PATH / "resources" / ".cache/sam")


# @pytest.fixture(scope="session")
# def model_sam_vit_b():
#     return get_model("vit_b", TESTS_PATH / "resources" / ".cache/sam")


# @pytest.fixture(scope="session")
# def label_config_sam():
#     with open(TESTS_PATH / "../annotation-templates/SAM-segmentation.yml") as f:
#         config = yaml.load(f, Loader=yaml.SafeLoader)
#     return config["config"]


# @pytest.fixture()
# def local_test_image(monkeypatch):

#     def mock_get_local_path(img_path, **kwargs):
#         return TESTS_PATH / "resources" / pathlib.Path(img_path).name

#     monkeypatch.setattr(
#         "label_studio_sdk._extensions.label_studio_tools.core.utils.io.get_local_path",
#         mock_get_local_path,
#     )
#     return None


@pytest.fixture(scope="function")
def patch_redis(redisdb, monkeypatch):

    def force_fakeredis(*args, **kwargs):
        return redisdb

    monkeypatch.setattr(
        "label_studio_berq.api.utils.get_redis_connection", force_fakeredis
    )


def test_get_rq_available_queues(redisdb, rqworker):
    with rqworker as _:
        queue_info = get_rq_available_queues(connection=redisdb)
        result = asyncio.run(queue_info)
    assert len(result) == 2


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
