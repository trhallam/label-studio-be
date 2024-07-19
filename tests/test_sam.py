import pytest
import os
import pathlib
import json

from models import SamModel

TESTS_PATH = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def sam_model(model_sam_vit_b, label_config_sam):
    # only load the model once
    os.environ["LSBE_MODEL_PATH"] = str(model_sam_vit_b["model_path"])
    os.environ["LSBE_SAM_MODEL_TYPE"] = "vit_b"
    model = SamModel(model_name="sam_vit_b", label_config=label_config_sam)
    return model


# def test_SamModel_init(model_sam_vit_h, label_config_sam):
#     os.environ["LSBE_MODEL_PATH"] = str(model_sam_vit_h["model_path"])
#     os.environ["LSBE_SAM_MODEL_TYPE"] = "vit_h"
#     model = SamModel(model_name="sam_vit_h", label_config=label_config_sam)


def test_sam_predict_keypoints(
    sam_model: SamModel, local_test_image, context_task, context_kp
):
    results = sam_model.predict(context_task, context_kp)
    results = sam_model.predict(context_task, context_kp)

    # check the response is jsonable
    json.dumps(results.model_dump())


def test_sam_predict_rectangles(
    sam_model: SamModel, local_test_image, context_task, context_rect
):
    results = sam_model.predict(context_task, context_rect)
    results = sam_model.predict(context_task, context_rect)

    # check the response is jsonable
    json.dumps(results.model_dump())
