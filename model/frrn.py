from .units import ConvUnit, FullResolutionResidualUnit, ResidualUnit

from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Activation, concatenate, Input
from keras.models import Model


class FullResolutionResidualNetworkBase(object):
    _model = None

    def model(self):
        return self._model


class FullResolutionResidualNetwork_A(FullResolutionResidualNetworkBase):
    def __init__(self, input_size, classes, *args, **kwargs):

        width, height = input_size

        input_layer = Input(shape=(height, width, 3))

        l = ConvUnit(input_layer)

        for i in range(3):
            l = ResidualUnit(l, 48)

        rs = Conv2D(32, (1, 1), padding='same')(l)
        ps = MaxPool2D()(l)

        for i in range(3):
            rs, ps = FullResolutionResidualUnit(rs, ps, 96, 2)

        ps = MaxPool2D()(ps)

        for i in range(4):
            rs, ps = FullResolutionResidualUnit(rs, ps, 192, 4)

        ps = MaxPool2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 384, 8)

        ps = MaxPool2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 384, 16)

        ps = UpSampling2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 192, 8)

        ps = UpSampling2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 192, 4)

        ps = UpSampling2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 96, 2)

        ps = UpSampling2D()(ps)

        x = concatenate([rs, ps], 3)

        for i in range(3):
            x = ResidualUnit(x, 48)

        x = Conv2D(classes, (1, 1))(x)
        x = Activation('sigmoid')(x)

        self._model = Model(inputs=input_layer, outputs=x)


class FullResolutionResidualNetwork_B(FullResolutionResidualNetworkBase):
    def __init__(self, input_size, classes, *args, **kwargs):

        width, height = input_size

        input_layer = Input(shape=(height, width, 3))

        l = ConvUnit(input_layer)

        for i in range(3):
            l = ResidualUnit(l, 48)

        rs = Conv2D(32, (1, 1), padding='same')(l)
        ps = MaxPool2D()(l)

        for i in range(3):
            rs, ps = FullResolutionResidualUnit(rs, ps, 96, 2)

        ps = MaxPool2D()(ps)

        for i in range(4):
            rs, ps = FullResolutionResidualUnit(rs, ps, 192, 4)

        ps = MaxPool2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 384, 8)

        ps = MaxPool2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 384, 16)

        # FRRN B features additional layers
        ps = MaxPool2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 384, 32)

        ps = UpSampling2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 192, 16)

        ps = UpSampling2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 192, 8)

        ps = UpSampling2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 192, 4)

        ps = UpSampling2D()(ps)

        for i in range(2):
            rs, ps = FullResolutionResidualUnit(rs, ps, 96, 2)

        ps = UpSampling2D()(ps)

        x = concatenate([rs, ps], 3)

        for i in range(3):
            x = ResidualUnit(x, 48)

        x = Conv2D(classes, (1, 1))(x)
        x = Activation('sigmoid')(x)

        self._model = Model(inputs=input_layer, outputs=x)


FULL_RESOLUTION_RESIDUAL_NETWORKS = {
    "A": FullResolutionResidualNetwork_A,
    "B": FullResolutionResidualNetwork_B
}
