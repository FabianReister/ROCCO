from .task import Task

from utils.preprocessor import Preprocessor
from datasets import DATASETS

from cv2 import imread, imwrite, resize, INTER_NEAREST
import numpy as np
import os.path

class TaskEvaluate(Task):
    def __init__(self, config):
        self._config = config

    @staticmethod
    def name():
        return "evaluate"

    def trainLabelToEvalFilename(self, label_filename, eval_dir):
        # /foo/bar/datasets/cityscapes/gtFine/test/aachen/aachen_000000_000019_gtFine_labelIds.png

        # expect the train image to have a test parent folder
        idx = label_filename.find('test')

        assert idx is not -1

        idx += 4 + 1  # length of 'test' + 1

        relative_label_filename = label_filename[idx:]

        return os.path.join(eval_dir, relative_label_filename)

    def run(self, model, log_dir):
        test_dataset = DATASETS[self._config["dataset"]["name"]](self._config["dataset"], 'test', loop=False)

        preprocessor = Preprocessor(self._config["preprocessor"])

        from evaluation import EVALUATORS
        evaluator = EVALUATORS[self._config["evaluator"]]

        gt_filenames = []
        prediction_filenames = []

        for img_filename, label_filename in test_dataset:
            prediction_filename = self.trainLabelToEvalFilename(label_filename, self._config["eval_dir"])

            # only run prediction if prediction image does not exist yet
            if not os.path.exists(prediction_filename):

                prediction_dir = os.path.dirname(prediction_filename)
                if not os.path.exists(prediction_dir):
                    os.makedirs(prediction_dir)

                img = imread(img_filename)

                assert img is not None

                # feed the image through the network
                x = preprocessor.process(img)
                y_pred = model.predict(np.asarray([x])).squeeze()
                y_label_pred = np.argmax(y_pred, axis=2)
                y_label_pred = np.asarray(y_label_pred, dtype=np.uint8)

                y_label_pred = resize(y_label_pred, (img.shape[1], img.shape[0]), interpolation=INTER_NEAREST)

                # store it in the eval folder
                imwrite(prediction_filename, y_label_pred)

            gt_filenames.append(label_filename)
            prediction_filenames.append(prediction_filename)

        evaluator.run(prediction_filenames, gt_filenames)