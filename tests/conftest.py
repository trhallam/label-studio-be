import pytest
import pathlib

import redis.exceptions
from rq import Queue, Worker
import redis

TESTS_PATH = pathlib.Path(__file__).parent

from tests.fixtures_resources import *


def redis_healthcheck(host: str, port: int):
    try:
        redisdb = redis.Redis(host=host, port=port)
        redisdb.ping()
        return True
    except redis.exceptions.ConnectionError:
        return False


def worker_healthcheck(host: str, port: int):
    try:
        redisdb = redis.Redis(host=host, port=port)
        worker_keys = Worker.all(redisdb)
        if worker_keys:
            return True
    except:
        pass
    return False


@pytest.fixture(scope="session")
def test_services(docker_ip, docker_services):
    """Start docker test services:
    - redis: A valkey redis server
    - worker: A basic worker instance
    """
    port = docker_services.port_for("redis", 6379)

    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: redis_healthcheck(docker_ip, port)
    )
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: worker_healthcheck(docker_ip, port)
    )

    redis_external = redis.Redis(host=docker_ip, port=port)
    workers = Worker.all(redis_external)

    return {"redisdb": redis_external, "workers": workers}


@pytest.fixture(scope="function")
def patch_redis(test_services, monkeypatch):

    def force_pytest_redis(*args, **kwargs):
        return test_services["redisdb"]

    monkeypatch.setattr(
        "label_studio_berq.api.utils.get_redis_connection", force_pytest_redis
    )


@pytest.fixture()
def rqqueue(test_services):
    redisdb = test_services["redisdb"]
    queue1 = Queue(name="pytest1", connection=redisdb)
    queue2 = Queue(name="pytest2", connection=redisdb)
    return [queue1, queue2]


@pytest.fixture()
def lsproject_json():
    return """{
        "project": "3.111",
        "schema": "<View></View>",
        "hostname": "http://localhost:8080",
        "access_token": "82d2dedc2777c40db97f4cc33a81d804886d96f1",
        "extra_params": "{\'a\':\'a\', \'b\':2.0}"
    }"""


@pytest.fixture()
def lsproject_setup(test_services, lsproject_json):
    redisdb = test_services["redisdb"]
    redisdb.hset(
        f"lsberq:project:3.111",
        "setup",
        value=lsproject_json,
    )
    return None
