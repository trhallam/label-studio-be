from rq.job import Job
from rq.queue import Queue
from rq.worker import Worker, WorkerStatus


class SAMWorker(Worker):

    redis_worker_namespace_prefix = "rq:worker:sam:"

    def execute_job(self, job: Job, queue: Queue):
        print(job)
        self.set_state(WorkerStatus.IDLE)


if __name__ == "__main__":
    from redis import Redis
    from rq import Queue

    redis = Redis(host="0.0.0.0", port=6379)
    queue = Queue("predict", connection=redis)

    worker = SAMWorker([queue], connection=redis)
    worker.work()
