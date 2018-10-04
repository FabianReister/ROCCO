from dataset import Dataset
import os


class EndoscopicInstrumentSegmentation(Dataset):
    """

    See
    http://opencas.webarchiv.kit.edu/?q=InstrumentCrowd

    Download link
    http://opencas.webarchiv.kit.edu/data/AnnotatedImages.zip

    """

    def __init__(self, config, type, loop=True):
        Dataset.__init__(self, loop)
        self._config = config

        dataset_path = config["path"]

        # if the dataset does not exist yet, download and extract it
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            self.download_and_extract_dataset()

        # get list of training files
        from glob2 import glob

        img_paths = r"{}/**/*_img[0-9][0-9].bmp".format(dataset_path)
        # [a-zA-Z0-9]+_img[0-9]+
        img_files = glob(img_paths)

        assert (len(img_files) > 0)

        print("Found %d images" % len(img_files))

        label_files = map(self.img_filename_to_label_filename, img_files)

        # check if any of the label files does not exist
        try:
            [os.stat(f) for f in label_files]
        except os.error:
            print("The label file %s does not exist" % f)

        self._dataset = zip(img_files, label_files)

    @staticmethod
    def name():
        return "endoscopic_instrument_segmentation"

    def download_and_extract_dataset(self):
        dataset_path = self._config["path"]

        # download the dataset
        print("Downloading dataset ...")

        DATASET_URI = "http://opencas.webarchiv.kit.edu/data/AnnotatedImages.zip"

        import urllib2
        dataset_zip = urllib2.urlopen(DATASET_URI)
        output_zip_filename = os.path.join(dataset_path, "AnnotatedImages.zip")

        with open(output_zip_filename, 'wb') as output:
            output.write(dataset_zip.read())

        print("Done downloading dataset.")
        print("Extracting ... ")

        import zipfile
        zip_ref = zipfile.ZipFile(output_zip_filename, 'r')
        zip_ref.extractall(dataset_path)
        zip_ref.close()

    def suffix_for_label_filename(self, img_filename):
        EXPERTS_SUFFIX = "_inst_GT.bmp"
        CROWD_SUFFIX = "_inst_GTcrowd.bmp"

        import re
        regexp = re.compile(r'crowd')
        if regexp.search(img_filename):
            return CROWD_SUFFIX
        else:
            return EXPERTS_SUFFIX

    def img_filename_to_label_filename(self, img_filename):
        suffix = self.suffix_for_label_filename(img_filename)
        return img_filename[:-4] + suffix

    def to_categorical(self, label_img):
        if len(label_img.shape) == 3:  # color image
            # grab the red channel where background is 0 and foreground 255
            label_img = label_img[0, :, :].ravel()

        # binary image. set foreground to 1
        label_img[label_img >= 1] = 1

        from keras.utils import to_categorical
        return to_categorical(label_img)