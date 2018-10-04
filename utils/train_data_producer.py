from cv2 import imread, IMREAD_ANYCOLOR, IMREAD_ANYDEPTH, IMREAD_GRAYSCALE
from numpy import asarray

class TrainDataProducer:
    def __init__(self, config, dataset, augmenter, preprocessor):
        self._config = config
        self._dataset = dataset
        self._augmenter = augmenter
        self._preprocessor = preprocessor

    def __iter__(self):
        return self

    def load_next_img_and_label(self):
        img_filename, label_filename = next(self._dataset)

        img = imread(img_filename, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH)
        label = imread(label_filename, IMREAD_GRAYSCALE)

        return img, label

    def next(self):
        X = []
        Y = []

        for i in range(self._config["batch_size"]):
            img, label = self.load_next_img_and_label()
            x, y = self._preprocessor.process(img, label)
            #x, y = self._augmenter.augment(x, y)

            y = self._dataset.to_categorical(y)

            #import numpy as np
            #y = np.expand_dims(y, 3)

            X.append(x)
            Y.append(y) # need to add first index (channel)

        return asarray(X), asarray(Y)

