from .dataset import Dataset


class Cityscapes(Dataset):
    def __init__(self, config):
        self._config = config

    @staticmethod
    def name():
        return "cityscapes"

