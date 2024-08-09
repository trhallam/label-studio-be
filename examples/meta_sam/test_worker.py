from typing import Iterator

import pytest
from pytest_docker.plugin import containers_scope, get_docker_services, Services

import pathlib
import redis
from rq import Queue, Worker

from label_studio_berq.test import worker_health_check

EXAMPLE_PATH = pathlib.Path(__file__).parent
DOCKER_COMPOSE_FILE = EXAMPLE_PATH / "docker-compose.yml"


@pytest.fixture(scope=containers_scope)
def docker_services_sam(
    docker_services,
    test_services,
    docker_compose_command: str,
    docker_compose_project_name: str,
    docker_setup: str,
    docker_cleanup: str,
) -> Iterator[Services]:
    """Start a docker compose project for the Sam Example

    This is just a single worker that piggybacks the main fixture (inc redis)
    from the project tests. The installation of pytorch is quite big, so it takes
    a while.
    """
    with get_docker_services(
        docker_compose_command,
        DOCKER_COMPOSE_FILE,
        docker_compose_project_name,
        docker_setup,
        docker_cleanup,
    ) as docker_service_sam:
        yield docker_service_sam


@pytest.fixture()
def samworker(test_services, docker_services, docker_services_sam, docker_ip):
    # port = docker_services.port_for("redis", 6379)

    redis_external = test_services["redisdb"]
    port = redis_external.connection_pool.connection_kwargs["port"]

    q = Queue("test-sam", connection=redis_external)
    docker_services_sam.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: worker_health_check(docker_ip, port, q)
    )
    workers = Worker.all(connection=redis_external, queue=q)
    return {"samworkers": workers}


def test_samworker_start(samworker):
    assert len(samworker["samworkers"]) == 1
