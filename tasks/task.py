from abc import abstractmethod


class Task(object):
    @abstractmethod
    def run(self, model):
        pass

    @staticmethod
    @abstractmethod
    def name():
        pass
