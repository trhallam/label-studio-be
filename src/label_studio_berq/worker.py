from typing import Dict, List, Optional
import logging

from rq import Queue
from rq.job import Job
from rq.logutils import setup_loghandlers
from rq.worker import SimpleWorker

from label_studio_ml.response import ModelResponse

from .api.models import SetupModel

LOGGER_NAME = "berq:Worker"
logger = logging.getLogger(LOGGER_NAME)


class LabelStudioBEWorker(SimpleWorker):

    redis_worker_namespace_prefix = "rq:worker:lsbe:"
    redis_lsberq_project_prefix = "lsberq:project:"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setup_loghandlers(name=LOGGER_NAME)

    def setup_model(self, *args, **kwargs):
        """Setup your model here. This should be called in the
        sub-classed __init__
        """
        # self.model = ?
        raise NotImplementedError

    def get_worker_func(self, func_name: str) -> callable:
        """Returns a function of the class instance from it's name.

        Args:
            func_name: The name of the class/instance function to return.

        Returns:
            self.func
        """
        return self.__getattribute__(func_name)

    def get_model_version(self):
        """Get the model version or return '0.0.1'"""
        try:
            model_version = self.model_version
        except AttributeError:
            model_version = "0.0.1"
        return model_version

    def get_project_setup(self, project: str) -> SetupModel:
        """Get the project setup details which are not sent with each normal
        worker request by label studio."""
        project_key = f"{self.redis_lsberq_project_prefix}{project}"
        return SetupModel.model_validate_json(
            self.connection.hget(project_key, "setup")
        )

    def get_project_setup_json(self, project: str) -> str:
        """Return the project setup as a json string, used for testing and validation"""
        setup = self.get_project_setup(project)
        return setup.model_dump_json()

    def predict(
        self, project: str, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Dispatches work to the model prediction logic

        Args:
            project: Label Studio project ID
            tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)

        Returns:
            model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        setup = self.get_project_setup(project)

        logger.debug(str(setup))

        model = getattr(self, "model", None)

        predictions = []
        if model is None:
            logger.debug("No model found in worker")
        elif setup is None:
            logger.debug("Could not find project setup")
        elif not tasks:
            logger.debug("No tasks to run")
        elif context and context.get("result"):
            # the user has provided context, so we are in prompt based prediction
            logger.debug("Prompt Prediction")
            predictions = model.prompt_predict(tasks, context=context, **kwargs)
        elif tasks and context == None:
            logger.debug("Pre-annotation")
            # apply auto segmentation to input tasks and return many predictions
            predictions = model.auto_predict(tasks, **kwargs)

        response = ModelResponse(predictions=predictions)
        # logger.debug(f"{response.model_dump()}")
        return response

    def fit(self, *args, **kwargs) -> Dict:
        raise NotImplementedError

    def execute_job(self, job: Job, queue: Queue):
        #     print(job)

        # patch the job to find the func_name in the worker instance
        job._deserialize_data()
        job_func = self.get_worker_func(job.func_name)
        job._instance = job_func.__self__
        job._func_name = job_func.__name__
        job.description = job.get_call_string()

        super().execute_job(job, queue)
