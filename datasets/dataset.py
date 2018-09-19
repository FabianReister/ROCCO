from abc import abstractmethod


class Dataset(object):

    @staticmethod
    @abstractmethod
    def name():
        pass

    @abstractmethod
    def next(self):
        pass

