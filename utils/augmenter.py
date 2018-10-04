import imgaug as ia
import imgaug.augmenters as iaa


class Augmenter:
    def __init__(self, config):
        self._config = config

        self._seq = iaa.Sequential([

            iaa.Sometimes(config["crop_probability"],
                          iaa.Crop(px=(0, 16))),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Sometimes(config["blur_probability"],
                          iaa.GaussianBlur(sigma=(0, 3.0))),  # blur images with a sigma of 0 to 3.0,

            iaa.Sometimes(config["affine_probability"], iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            ))
        ])

        # https://github.com/aleju/imgaug/issues/41
        self._hooks_semantic = ia.HooksImages(activator=self.activator_semantic)

    @staticmethod
    def activator_semantic(images, augmenter, parents, default):
        whitelist = ["Affine", "Crop"]

        if augmenter.name in whitelist:
            return default
        else:
            return False

    def augment(self, x, y):

        # x and y augmentation has to be in sync
        self._seq.to_deterministic()

        x_aug = self._seq.augment_images(x)

        # augmenter expects image to have format (width, height, channels)
        if len(y.shape) == 2:
            import numpy as np
            y = np.expand_dims(y, 3)

        y_aug = self._seq.augment_images(y, hooks=self._hooks_semantic)

        return x_aug, y_aug.squeeze()
