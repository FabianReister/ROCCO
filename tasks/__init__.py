from .task import Task

# import all subclasses
from .train import TaskTrain
from .predict import TaskPredict

TASKS = dict([(task.name(), task) for task in Task.__subclasses__()])
