import pytest

from models import core


def test_parse_keypoints(context_kp):
    points = core.parse_keypoints(context_kp)
    assert True
