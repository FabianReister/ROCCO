import unittest

from datasets.cityscapes import Cityscapes

class TestCityscapes(unittest.TestCase):

    def test_filename_conversion(self):

        img_filename = "datasets/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
        label_filename = "datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png"

        self.assertEqual(Cityscapes.img_to_label_filename(img_filename), label_filename)


if __name__ == '__main__':
    unittest.main()