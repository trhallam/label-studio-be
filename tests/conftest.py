import pytest
import json
import yaml
import pathlib

import label_studio_sdk._extensions.label_studio_tools.core.utils.io

TESTS_PATH = pathlib.Path(__file__).parent

from models.sam.model import get_model


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
def model_sam_vit_h():
    return get_model("vit_h", TESTS_PATH / "resources" / ".cache/sam")


@pytest.fixture(scope="session")
def model_sam_vit_b():
    return get_model("vit_b", TESTS_PATH / "resources" / ".cache/sam")


@pytest.fixture(scope="session")
def label_config_sam():
    with open(TESTS_PATH / "../annotation-templates/SAM-segmentation.yml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config["config"]


@pytest.fixture()
def local_test_image(monkeypatch):

    def mock_get_local_path(img_path, **kwargs):
        return TESTS_PATH / "resources" / pathlib.Path(img_path).name

    monkeypatch.setattr(
        "label_studio_sdk._extensions.label_studio_tools.core.utils.io.get_local_path",
        mock_get_local_path,
    )
    return None
