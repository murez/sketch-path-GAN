from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import *
from keras.models import *
from keras.initializers import RandomNormal
import keras.backend as K
import tensorflow as tf


class InstanceNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-7, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(
                self.axis) + ' of input tensor should have a defined dimension but the layer received an input with shape ' + str(
                input_shape) + '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer='ones')
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)


def ReflectPad(x, ks=1):
    return tf.pad(x, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * K.elu(x))


def ResBlock(y, dim):
    x = Lambda(ReflectPad)(y)
    x = Conv2D(dim, 3, padding='valid')(x)
    x = LeakyReLU()(x)
    x = Lambda(ReflectPad)(x)
    x = Conv2D(dim, 3, padding='valid')(x)
    x = LeakyReLU()(x)
    return add([x, y])


def ResizeConv(x, cdim):
    def Resize2x(x):
        shape = x.get_shape()
        new_shape = tf.shape(x)[1:3] * tf.constant(np.array([2, 2], dtype='int32'))
        x = tf.image.resize_bilinear(x, new_shape)
        x.set_shape((None, shape[1] * 2, shape[2] * 2, None))
        return x

    x = Lambda(Resize2x)(x)
    x = Lambda(lambda x: ReflectPad(x, 1))(x)
    x = Conv2D(cdim, 3, padding='valid')(x)
    return x


def BuildGenerator(img):
    gdim = 32
    x = Lambda(lambda x: ReflectPad(x, 3))(img)
    x = Conv2D(gdim, 7, strides=1, padding='valid')(x)
    x = LeakyReLU()(x)
    x = Lambda(lambda x: ReflectPad(x, 1))(x)
    x = Conv2D(gdim * 2, 3, strides=2, padding='valid')(x)
    x = LeakyReLU()(x)
    x = Lambda(lambda x: ReflectPad(x, 1))(x)
    x = Conv2D(gdim * 4, 3, strides=2, padding='valid')(x)
    x = LeakyReLU()(x)
    for ii in range(9): x = ResBlock(x, gdim * 4)
    x = ResizeConv(x, gdim * 2)
    x = LeakyReLU()(x)
    x = ResizeConv(x, gdim)
    x = LeakyReLU()(x)
    x = Lambda(lambda x: ReflectPad(x, 1))(x)
    x = Conv2D(3, 3, strides=1, padding='valid', activation='tanh')(x)
    return Model(inputs=img, outputs=x)


def BuildDiscriminator(img):
    ddim = 64
    x = Conv2D(ddim, 7, padding='same', strides=2)(img)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)
    # x = Dropout(.5)(x)
    x = Conv2D(ddim * 2, 3, padding='same', strides=2)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)
    # x = Dropout(.5)(x)
    x = Conv2D(ddim * 4, 3, padding='same', strides=2)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)
    # x = Dropout(.5)(x)
    x = Conv2D(ddim * 8, 3, padding='same', strides=2)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)
    # x = Dropout(.5)(x)
    x = Conv2D(1, 3, padding='same', strides=1)(x)
    return Model(inputs=img, outputs=x)
