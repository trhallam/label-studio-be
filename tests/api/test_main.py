import pytest
from fastapi.testclient import TestClient

from label_studio_berq.api.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "label-studio-be API Server"}


def test_health(rq_test_queues, test_services, patch_redis):
    for queue in rq_test_queues.queues:
        response = client.get(f"/{queue}/health")
        content = response.json()
        assert content["MODEL_CLASS"] == queue
        assert content["status"] == "UP"


def test_status(rq_test_queues, test_services, patch_redis):
    response = client.get("/status")
    worker_info = response.json()
    assert len(worker_info) == 1
    key, values = worker_info.popitem()
    assert values["queues"] == "pytest1,pytest2"


def test_rq_queues(test_services, patch_redis):
    response = client.get("/rq-queues")
    queue_info = response.json()
    assert len(queue_info) == 2


def test_setup(rq_test_queues, test_services, patch_redis):
    for q in rq_test_queues.queues:
        response = client.post(
            f"/{q}/setup",
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
        assert content["model_version"] == "0.0.1"


def test_predict_kp(test_services, patch_redis, predict_json):
    response = client.post("/pytest1/predict", json=predict_json)
