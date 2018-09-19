from .task import Task

from utils.train_data_producer import TrainDataProducer
from utils.augmenter import Augmenter
from utils.preprocessor import Preprocessor
from datasets import DATASETS


class TaskTrain(Task):
    def __init__(self, config):
        self._config = config

    @staticmethod
    def name():
        return "train"

    def run(self, model):
        dataset = DATASETS[self._config["dataset"]["name"]](self._config["dataset"])
        augmenter = Augmenter(self._config["augmenter"])
        preprocessor = Preprocessor(self._config["preprocessor"])

        from keras.callbacks import TensorBoard
        tensor_board = TensorBoard()

        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        train_data_producer = TrainDataProducer(config=self._config, dataset=dataset, augmenter=augmenter, preprocessor=preprocessor)

        model.fit_generator(generator=train_data_producer, steps_per_epoch=self._config["steps_per_epoch"],
                            epochs=self._config["epochs"], callbacks=[tensor_board])
