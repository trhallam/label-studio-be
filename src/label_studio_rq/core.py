def parse_keypoints(context: str, kp_id: str = "keypoint") -> dict:
    """Parse keypoints context to return points and labels."""

    f = filter(lambda x: x["type"] == kp_id, context["result"])
    return f


def parse_rectpoints(context: str, rect_id: str = "rect"):
    pass
