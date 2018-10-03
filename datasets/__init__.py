from .dataset import Dataset

# import all subclasses
from cityscapes import Cityscapes
from endoscopic_instrument_segmentation import EndoscopicInstrumentSegmentation

DATASETS = dict([(dataset.name(), dataset) for dataset in Dataset.__subclasses__()])
