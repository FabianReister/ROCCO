from cv2 import imread, IMREAD_ANYCOLOR, IMREAD_ANYDEPTH, IMREAD_GRAYSCALE
from numpy import asarray


class ValidDataProducer():
    def __init__(self, config, dataset, preprocessor):
        self._config = config
        self._dataset = dataset
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

            from keras.utils import to_categorical

            # The void class is 255 in this case. This will result in a very large categorical matrix.
            # Therefore set 255 to the next id after the last valid class, which is 20 in this case.
            y[y >= 20] = 19


            import numpy as np
            y = np.expand_dims(y, 3)

            y = to_categorical(y)

            X.append(x)
            Y.append(y) # need to add first index (channel)

        return asarray(X), asarray(Y)
