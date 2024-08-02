from rq import Queue
from rq.job import Job
from rq.worker import WorkerStatus, SimpleWorker, BaseWorker

from .api.models import SetupModel


class LabelStudioBEWorker(SimpleWorker):

    redis_worker_namespace_prefix = "rq:worker:lsbe:"
    redis_lsberq_project_prefix = "lsberq:project:"

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

    def predict(self):
        raise NotImplementedError

    def fit(self):
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
