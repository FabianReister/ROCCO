from .dataset import Dataset

# import all subclasses
from cityscapes import Cityscapes

DATASETS = dict([(dataset.name(), dataset) for dataset in Dataset.__subclasses__()])
