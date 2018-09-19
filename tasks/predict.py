from .task import Task


class TaskPredict(Task):
    def __init__(self, config):
        self._config = config

    @staticmethod
    def name():
        return "predict"

    def run(self, model):
        pass
