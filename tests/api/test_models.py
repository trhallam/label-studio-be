import pytest
import json

from label_studio_berq.api.models import SetupModel, PredictModel


def test_SetupModel():
    json_data = {
        "project": "3.1722346376",
        "schema": "<View></View>",
        "hostname": "http://localhost:8080",
        "access_token": "82d2dedc2777c40db97f4cc33a81d804886d96f1",
        "extra_params": "{'a':'a', 'b':2.0}",
    }
    SetupModel.model_validate_json(json.dumps(json_data), strict=True)
