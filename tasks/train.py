from .task import Task

from utils.train_data_producer import TrainDataProducer, ValidDataProducer
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
        train_dataset = DATASETS[self._config["dataset"]["name"]](self._config["dataset"], 'train')
        valid_dataset = DATASETS[self._config["dataset"]["name"]](self._config["dataset"], 'val')

        augmenter = Augmenter(self._config["augmenter"])
        preprocessor = Preprocessor(self._config["preprocessor"])

        from keras.callbacks import TensorBoard
        import datetime
        tensor_board = TensorBoard(log_dir="./logs/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        train_data_producer = TrainDataProducer(config=self._config, dataset=train_dataset, augmenter=augmenter,
                                                preprocessor=preprocessor)
        valid_data_producer = ValidDataProducer(config=self._config, dataset=valid_dataset, preprocessor=preprocessor)

        model.fit_generator(generator=train_data_producer, steps_per_epoch=self._config["steps_per_epoch"],
                            validation_data=valid_data_producer, validation_steps=self._config["validation_steps"],
                            epochs=self._config["epochs"], callbacks=[tensor_board])

        model.save('frrn.h5')
