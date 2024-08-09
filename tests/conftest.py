from typing import Dict, List
import pytest
import pathlib

from rq import Queue

TESTS_PATH = pathlib.Path(__file__).parent

from tests.fixtures_resources import *

pytest_plugins = ["label_studio_berq.test"]


@pytest.fixture()
def rqqueue(test_services: Dict) -> List[Queue]:
    redisdb = test_services["redisdb"]
    queue1 = Queue(name="pytest1", connection=redisdb)
    queue2 = Queue(name="pytest2", connection=redisdb)
    return [queue1, queue2]


@pytest.fixture()
def lsproject_json() -> str:
    return """{
        "project": "3.111",
        "schema": "<View></View>",
        "hostname": "http://localhost:8080",
        "access_token": "82d2dedc2777c40db97f4cc33a81d804886d96f1",
        "extra_params": "{\'a\':\'a\', \'b\':2.0}"
    }"""


@pytest.fixture()
def lsproject_setup(test_services, lsproject_json) -> None:
    redisdb = test_services["redisdb"]
    redisdb.hset(
        f"lsberq:project:3.111",
        "setup",
        value=lsproject_json,
    )
    return None
