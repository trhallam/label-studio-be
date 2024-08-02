from typing import Dict
from redis import Redis
from rq.worker_registration import get_keys as get_rq_worker_keys


def get_rq_worker_info(connection: Redis) -> Dict:
    """Get information on available rq workers from Redis

    Args:
        connection: Uses the env vars if connection is None.

    Returns:
        RQ worker information from Redis
    """
    worker_keys = get_rq_worker_keys(connection=connection)
    worker_info = {key: connection.hgetall(key) for key in worker_keys}
    return worker_info
