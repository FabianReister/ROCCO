from cv2 import imread, IMREAD_ANYCOLOR, IMREAD_ANYDEPTH, IMREAD_GRAYSCALE

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
        for i in range(self._config["batch_size"]):
            img, label = self.load_next_img_and_label()



