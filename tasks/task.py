from abc import abstractmethod


class Task(object):
    @abstractmethod
    def run(self, model, log_dir):
        pass

    @staticmethod
    @abstractmethod
    def name():
        pass
