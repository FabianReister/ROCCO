from .dataset import Dataset


class Cityscapes(Dataset):
    def __init__(self, config, type, loop=True):
        Dataset.__init__(self, loop)
        dataset_path = config["path"]
        self._config = config

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

    def to_categorical(self, label_img):
        if len(label_img.shape) == 3:  # color image
            from cv2 import cv2
            label_img = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)

        from keras.utils import to_categorical

        # The void class is 255 in this case. This will result in a very large categorical matrix.
        # Therefore set 255 to the next id after the last valid class, which is 20 in this case.
        num_classes = self._config["classes"]
        label_img[label_img >= num_classes] = num_classes - 1

        return to_categorical(label_img)

