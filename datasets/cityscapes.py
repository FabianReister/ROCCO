from .dataset import Dataset


class Cityscapes(Dataset):
    def __init__(self, config, type, loop=True):
        Dataset.__init__(self, loop)
        dataset_path = config["path"]

        img_paths = "{}/leftImg8bit/{}/*/*.png".format(dataset_path, type)

        from glob2 import glob
        img_files = glob(img_paths)

        label_files = map(self.img_to_label_filename, img_files)

        self._dataset = zip(img_files, label_files)

    @staticmethod
    def name():
        return "cityscapes"

    @staticmethod
    def img_to_label_filename(img_filename):
        img_filename = img_filename.replace('/leftImg8bit/', '/gtFine/')
        img_filename = img_filename.replace('leftImg8bit', 'gtFine_labelTrainIds')
        return img_filename


