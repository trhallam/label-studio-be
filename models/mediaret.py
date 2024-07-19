"""Media Retrieval 

This module contains functions and classes necessary for recovering data from storage.

It fills gaps in the Label Studio SDK.

"""

from typing import Union, List, Dict, Optional, Tuple, Callable
import os
import logging

from label_studio_ml.utils import InMemoryLRUDictCache

import label_studio_sdk._extensions.label_studio_tools.core.utils.io as lsio

LBSE_ACCESS_TOKEN = os.environ.get("LBSE_ACCESS_TOKEN")
LBSE_HOST = os.environ.get("LBSE_HOST")


def get_media_path(url: Union[str, os.PathLike], task):
    """ """
    media_path = lsio.get_local_path(
        url,
        access_token=LBSE_ACCESS_TOKEN,
        hostname=LBSE_HOST,
        task_id=task.get("id"),
    )
    return media_path


class MediaCache:

    logger = logging.getLogger("MediaCache")

    def __init__(self, n: int = 10):
        """
        Args:
            n: The number of items to cache
        """
        self.cache = InMemoryLRUDictCache(n)

    def _media_to_cache_name(self, media: Union[str, os.PathLike]) -> str:
        return str(media)

    def get_media(self, media: Union[str, os.PathLike], task=None):
        """
        Args:
            The media name/path/url/id
        """
        cached_payload = self.cache.get(self._media_to_cache_name(media))

        if cached_payload is None:
            self.logger.debug(f"Caching media: {media}")
            media = get_media_path(media, task)
            return None, media
        else:
            self.logger.debug(f"Using cached media: {media}")
            return cached_payload, None

    def cache_media(self, media: Union[str, os.PathLike], payload: Dict):
        """Cache some media for reuse.

        Args:
            media: The name of the in cache payload
            payload: Dictionary of cacheable objects
        """
        cache_name = self._media_to_cache_name(media)
        self.logger.debug(f"Storing payload {cache_name}")
        self.cache.put(cache_name, payload)
