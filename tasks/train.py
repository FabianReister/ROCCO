from .task import Task

from utils.train_data_producer import TrainDataProducer
from utils.validation_data_producer import ValidDataProducer
from utils.augmenter import Augmenter
from utils.preprocessor import Preprocessor
from datasets import DATASETS


class TaskTrain(Task):
    def __init__(self, config):
        self._config = config

    @staticmethod
    def name():
        return "train"

    def run(self, model, log_dir):


        train_dataset = DATASETS[self._config["dataset"]["name"]](self._config["dataset"], 'train')
        val_dataset = DATASETS[self._config["dataset"]["name"]](self._config["dataset"], 'val')

        augmenter = Augmenter(self._config["augmenter"])
        preprocessor = Preprocessor(self._config["preprocessor"])

        from keras.callbacks import TensorBoard
        tensor_board = TensorBoard(log_dir=log_dir)

        from keras.optimizers import get as get_optimizer
        optimizer = get_optimizer(self._config["optimizer"])

        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        train_data_producer = TrainDataProducer(config=self._config, dataset=train_dataset, augmenter=augmenter,
                                                preprocessor=preprocessor)
        valid_data_producer = ValidDataProducer(config=self._config, dataset=val_dataset, preprocessor=preprocessor)

        train_steps_per_epoch = train_dataset.num_samples() / self._config["batch_size"]
        val_steps_per_epoch = val_dataset.num_samples() / self._config["batch_size"]

        model.fit_generator(generator=train_data_producer,
                            steps_per_epoch=train_steps_per_epoch,
                            validation_data=valid_data_producer,
                            validation_steps=val_steps_per_epoch,
                            epochs=self._config["epochs"],
                            callbacks=[tensor_board])

        # save the model in the log directory
        import os
        trained_model_filename = os.path.join(log_dir, 'trained_model.h5')

        print("Saving trained model to %s" % trained_model_filename)

        # If the saving does not work, take a look at
        # https://github.com/keras-team/keras/issues/6766
        # and then upgrade keras!
        model.save(trained_model_filename)
