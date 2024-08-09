"""Fixtures to help for testing of package and examples

**How to use:**

Put the following in your test conftest.py

```python
pytest_plugins = ["label_studio_berq.test"]
```

"""

from typing import Union, List
from dataclasses import dataclass

import pytest
import redis

from rq import Worker, Queue


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("docker")
    group.addoption(
        "--docker-no-build",
        action="store_true",
        help="Disable building of docker images for tests.",
        default=False,
    )


@dataclass
class RqTestQueues:
    queues: List[str]
    monkey_patch: pytest.MonkeyPatch


@pytest.fixture(scope="session")
def rq_test_queues():
    queues = ["pytest1", "pytest2"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTEST_QUEUES", " ".join(queues))
        yield RqTestQueues(queues, mp)


@pytest.fixture(scope="session")
def docker_setup(pytestconfig: pytest.Config) -> Union[List[str], str]:
    """Get the docker_compose command to be executed for test setup actions.
    Override this fixture in your tests if you need to change setup actions.
    Returning anything that would evaluate to False will skip this command."""

    nobuild = pytestconfig.getoption("docker_no_build")

    if nobuild:
        return ["up -d"]

    return ["up --build -d"]


@pytest.fixture(scope="session", autouse=True)
def docker_compose_project_name() -> str:
    """Generate a project name using the current process PID. Override this
    fixture in your tests if you need a particular project name."""

    return "pytest-label-studio-berq"


def redis_health_check(host: str, port: int):
    try:
        redisdb = redis.Redis(host=host, port=port)
        redisdb.ping()
        return True
    except redis.exceptions.ConnectionError:
        return False


def worker_health_check(host: str, port: int, queue: Queue):
    try:
        redisdb = redis.Redis(host=host, port=port)
        worker_keys = Worker.all(redisdb, queue=queue)
        if worker_keys:
            return True
    except:
        pass
    return False


@pytest.fixture(scope="session")
def test_services(
    docker_compose_project_name, rq_test_queues, docker_ip, docker_services
):
    """Start docker test services:
    - redis: A valkey redis server
    - worker: A basic worker instance
    """
    port = docker_services.port_for("redis", 6379)
    redis_external = redis.Redis(host=docker_ip, port=port)
    queues = [Queue(q, connection=redis_external) for q in rq_test_queues.queues]
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: redis_health_check(docker_ip, port)
    )
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: worker_health_check(docker_ip, port, queues[0]),
    )

    workers = Worker.all(redis_external)

    return {"redisdb": redis_external, "workers": workers}


@pytest.fixture(scope="function")
def patch_redis(test_services, monkeypatch):

    def force_pytest_redis(*args, **kwargs):
        return test_services["redisdb"]

    monkeypatch.setattr(
        "label_studio_berq.api.utils.get_redis_connection", force_pytest_redis
    )
    connection_kwargs = test_services["redisdb"].connection_pool.connection_kwargs

    monkeypatch.setattr(
        "label_studio_berq.api.utils.LSBE_RQ_REDIS_HOST", connection_kwargs["host"]
    )
    monkeypatch.setattr(
        "label_studio_berq.api.utils.LSBE_RQ_REDIS_PORT", connection_kwargs["port"]
    )
