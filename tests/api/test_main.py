import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from label_studio_berq.api.main import app
from label_studio_berq.api.utils import get_rq_worker_info, get_rq_available_queues

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "label-studio-be API Server"}


def test_health(rqworker, patch_redis):
    with rqworker as _:
        response = client.get("/pytest1/health")
        content = response.json()
    assert content["MODEL_CLASS"] == "pytest1"
    assert content["status"] == "UP"


def test_status(rqworker, patch_redis):
    with rqworker as _:
        response = client.get("/status")
    worker_info = response.json()
    assert len(worker_info) == 1
    key, values = worker_info.popitem()
    assert values["queues"] == "pytest1,pytest2"


def test_rq_queues(rqworker, patch_redis):
    with rqworker as _:
        response = client.get("/rq-queues")
    queue_info = response.json()
    assert len(queue_info) == 2


def test_setup(redisdb, rqworker, fakeredis, patch_redis):
    with rqworker as _:
        import time

        time.sleep(2)
        response = client.post(
            "/pytest1/setup",
            json={
                "project": "3.1722346376",
                "schema": "<View></View>",
                "hostname": "http://localhost:8080",
                "access_token": "82d2dedc2777c40db97f4cc33a81d804886d96f1",
                "extra_params": {"a": "aaa", "b": 2.0},
            },
        )

    content = response.json()
    assert response.status_code == 200
    assert content["model_version"] == "0.1.0"


def test_predict_kp(redisdb, rqworker, fakeredis, patch_redis, context_kp):
    with rqworker as _:
        response = client.post("/pytest1/predict", json=context_kp)
