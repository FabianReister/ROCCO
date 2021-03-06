from .task import Task

# import all subclasses
from .train import TaskTrain
from .predict import TaskPredict
from .evaluate import TaskEvaluate

TASKS = dict([(task.name(), task) for task in Task.__subclasses__()])
