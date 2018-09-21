from abc import abstractmethod


class Dataset(object):
    _dataset = []
    _idx = -1

    def __init__(self, loop):
        self.__loop = loop

    def __iter__(self):
        return self

    @staticmethod
    @abstractmethod
    def name():
        pass

    def next(self):
        self._idx += 1

        # reset if required
        if self._idx >= len(self._dataset):
            if self.__loop is True:
                self._idx = 0
            else:
                raise StopIteration()

        return self._dataset[self._idx]

