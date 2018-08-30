from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import *
from keras.models import *
import keras.backend as K
import tensorflow as tf

#tf.InstanceNormalization


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


def BuildGenerator(img):
    x = Lambda(lambda x: ReflectPad(x, 3))(img)
    x = Conv2D(48, 5, strides=2, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 3, strides=2, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(1024, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(1024, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 4, strides=2, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(48, 3, strides=1, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(24, 4, strides=2, padding='valid')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Conv2D(1, 3, strides=1, padding='valid')(x)
    x = Softmax()(x)
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
