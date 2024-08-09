from typing import List, Type, Union, Dict, Optional
import pathlib
import click
import logging

from rq.queue import Queue
from rq.logutils import setup_loghandlers
from redis import Redis

import typer
from typing_extensions import Annotated

from label_studio_ml.response import ModelResponse
from label_studio_berq.worker import LabelStudioBEWorker
from label_studio_berq.typer import DomainIpParamType, DomainIpParser

from model import SamModel, MODEL_TYPE_CHOICES


LOGGING_FMT = "%(asctime)s %(name)s %(levelname)s %(message)s"
LOGGER_NAME = "berq.SAMWorker"
logger = logging.getLogger(LOGGER_NAME)


class SAMWorker(LabelStudioBEWorker):

    redis_worker_namespace_prefix = "rq:worker:sam:"
    model_version = "0.1.0"

    def __init__(self, model_path, model_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setup_loghandlers(
            log_format=LOGGING_FMT,
            name=LOGGER_NAME,
        )
        self.model_path = model_path
        self.model_type = model_type
        self.setup_model()

    def setup_model(self):
        logger.info("Setting up Model")
        self.model = SamModel(self.model_path, self.model_type)
        logger.debug("Model ready")


def main(
    model_path: Annotated[
        pathlib.Path, typer.Option(help="Path to the SAM model checkpoint")
    ],
    model_type: Annotated[
        str,
        typer.Option(
            click_type=click.Choice(MODEL_TYPE_CHOICES),
            help="The SAM model checkpoint type",
        ),
    ],
    queue: Annotated[List[str], typer.Argument(help="RQ queue/s")],
    redis_host: Annotated[
        DomainIpParamType,
        typer.Option(click_type=DomainIpParser(), help="The redis server address"),
    ] = "http://0.0.0.0",
    port: Annotated[int, typer.Option(help="The redis server port")] = 6379,
    log_level: Annotated[
        str,
        typer.Option(
            help="Logging level",
            click_type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
        ),
    ] = None,
):
    """
    Start the SAMWorker and join the RQ `queue`.
    """
    # setup logging level
    if log_level:
        logger.setLevel(log_level)

    redis = Redis(host=str(redis_host), port=port)
    queues = [Queue(q, connection=redis) for q in queue]

    worker = SAMWorker(model_path, model_type, queues, connection=redis)
    worker.work(logging_level=log_level, log_format=LOGGING_FMT)


if __name__ == "__main__":
    typer.run(main)
