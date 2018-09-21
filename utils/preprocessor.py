from cv2 import resize, INTER_NEAREST, INTER_LINEAR
from numpy import asarray


class Preprocessor:
    def __init__(self, config):
        self._config = config

    def process(self, img, label=None):
        img = self._resize(img, INTER_LINEAR)
        img = self._normalize(img)

        if label is not None:
            label = self._resize(label, INTER_NEAREST)
            return img, label

        return img

    def _resize(self, img, type):
        width, height = self._config["input_size"]
        return resize(img, (width, height), type)

    @staticmethod
    def _normalize(img):
        return asarray(img, dtype=float) / 255.0
