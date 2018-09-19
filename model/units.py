from keras.layers import Conv2D, BatchNormalization, Activation, concatenate, UpSampling2D, MaxPool2D, add

"""

Here, the building blocks of a Full Resolution Residual Network are defined.

"""

def ConvUnit(input):
    """
    the first layers of the network
    :param input:
    :return:
    """
    x = Conv2D(48, (5, 5), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def ResidualUnit(input, n_kernels):
    channels_in = input.shape[3]
    channels_out = n_kernels

    # If the number of input channels is different from the number of output
    # channels, then we have to add a linear projection
    if channels_in != channels_out:
        input = Conv2D(channels_out, (1, 1), use_bias=False)(input)

    x = Conv2D(n_kernels, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(n_kernels, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = add([input, x])

    return x


def FullResolutionResidualUnit(z, y, n_kernels, pooling_size):
    g = MaxPool2D(pool_size=(pooling_size, pooling_size))(z)
    g = concatenate([g, y], 3)

    g = Conv2D(n_kernels, (3, 3), padding='same')(g)
    g = BatchNormalization()(g)
    g = Activation('relu')(g)

    g = Conv2D(n_kernels, (3, 3), padding='same')(g)
    g = BatchNormalization()(g)
    g = Activation('relu')(g)
    y = g

    h = Conv2D(32, (1, 1), padding='same')(g)
    h = UpSampling2D(size=(pooling_size, pooling_size))(h)

    z = add([z, h])

    return z, y
