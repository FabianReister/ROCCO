from abc import abstractmethod


class Evaluator(object):
    def __init__(self, log_dir):
        self._log_dir = log_dir

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, prediction_filenames, ground_truth_filenames):
        """
        Runs the evaluation on a set of files

        :param prediction_filenames:
        :param ground_truth_filenames:
        :return:
        """
        pass


class EvaluatorCityscapes(Evaluator):
    def __init__(self, *args, **kwargs):
        Evaluator.__init__(*args, **kwargs)

    def name(self):
        return "cityscapes"

    def run(self, prediction_filenames, ground_truth_filenames):
        # finally run evaluation
        from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import evaluateImgLists
        args = {
            'quiet': True
        }
        confusion_matrix = evaluateImgLists(prediction_filenames, ground_truth_filenames, args)

        import numpy as np
        np.savetxt("{}/confusion_matrix.csv".format(self._log_dir), confusion_matrix)