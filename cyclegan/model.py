from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import *
from keras.models import *
import keras.backend as K
import tensorflow as tf

tf.InstanceNormalization


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


def BuildGenerator():
    def Generator():
        ModelG = Sequential()
        # input picture
        ModelG.add(Dense())
        # down laying
        ModelG.add(Conv2D(48, 5, strides=2, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(128, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(128, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(256, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(256, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(256, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(256, 3, strides=2, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(512, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(1024, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        # flat laying
        ModelG.add(Conv2D(1024, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(1024, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(1024, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(512, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(256, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        # up laying
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(256, 4, strides=2, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(256, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(128, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(128, 4, strides=2, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(128, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(48, 3, strides=1, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(24, 4, strides=2, padding='valid'))
        ModelG.add(BatchNormalization(axis=1))
        ModelG.add(LeakyReLU())
        ModelG.add(Conv2D(1, 3, strides=1, padding='valid', activation="Sigmoid"))
        return ModelG


def BuildDiscriminator():
    ModelB = Sequential()
    ModelB.add(Conv2D(64, 7, padding='same', strides=2))
    ModelB.add(InstanceNormalization())
    ModelB.add(LeakyReLU())
    ModelB.add(Conv2D(128, 3, padding='same', strides=2))
    ModelB.add(InstanceNormalization())
    ModelB.add(LeakyReLU())
    ModelB.add(Conv2D(256, 3, padding='same', strides=2))
    ModelB.add(InstanceNormalization())
    ModelB.add(LeakyReLU())
    ModelB.add(Conv2D(512, 3, padding='same', strides=2))
    ModelB.add(InstanceNormalization())
    ModelB.add(LeakyReLU())
    ModelB.add(Conv2D(1, 3, padding='same', strides=2))
    return ModelB
