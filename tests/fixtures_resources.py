import pytest
import pathlib
import json

TESTS_PATH = pathlib.Path(__file__).parent


@pytest.fixture(scope="session")
def context_kp():
    with open(TESTS_PATH / "resources" / "context-kp2.json") as f:
        doc = json.load(f)
    return doc


@pytest.fixture(scope="session")
def context_rect():
    with open(TESTS_PATH / "resources" / "context-rect.json") as f:
        doc = json.load(f)
    return doc


@pytest.fixture(scope="session")
def context_task():
    with open(TESTS_PATH / "resources" / "task.json") as f:
        doc = json.load(f)
    return doc


@pytest.fixture(scope="session")
def context_preanno():
    with open(TESTS_PATH / "resources" / "context-preanno.json") as f:
        doc = json.load(f)
    return doc


@pytest.fixture(scope="session")
def predict_json():
    with open(TESTS_PATH / "resources" / "predict.json") as f:
        doc = json.load(f)
    return doc
