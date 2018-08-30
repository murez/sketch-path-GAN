#ljqpy,
import os, sys, time, random
import numpy as np
import keras.backend as K
from collections import defaultdict
from keras.datasets import mnist
from keras.utils.generic_utils import Progbar
from keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from keras.preprocessing import image
from PIL import Image

time.clock()

np.random.seed(1333)
K.set_image_dim_ordering('tf')

from .model import BuildGenerator, BuildDiscriminator

# params
nb_epochs = 200
batch_size = 1
p_lambda = 10

imgsize = 256

adam_lr = 0.000005
adam_beta_1 = 0.5

imgdirA = '/mnt/smb25/ImageNet/horse2zebra/trainA'
imgdirB = '/mnt/smb25/ImageNet/horse2zebra/trainB'
testimgdirA = '/mnt/smb25/ImageNet/horse2zebra/testA'
testimgdirB = '/mnt/smb25/ImageNet/horse2zebra/testB'

modeldir = 'data/'
testimgdir = 'images/'

modelG = BuildGenerator(Input(shape=(imgsize, imgsize, 3)))
modelF = BuildGenerator(Input(shape=(imgsize, imgsize, 3)))
modelDG = BuildDiscriminator(Input(shape=(imgsize, imgsize, 3)))
modelDF = BuildDiscriminator(Input(shape=(imgsize, imgsize, 3)))

modelG.summary()
modelDG.summary()

try:
    modelG.load_weights(os.path.join(modeldir, 'modelG.h5'))
    modelF.load_weights(os.path.join(modeldir, 'modelF.h5'))
    modelDG.load_weights(os.path.join(modeldir, 'modelDG.h5'))
    modelDF.load_weights(os.path.join(modeldir, 'modelDF.h5'))
except Exception as e:
    print(e)

modelDG.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')
modelDF.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')
modelG.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')
modelF.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')

imageReal = Input(shape=(imgsize, imgsize, 3))
imageFake = Input(shape=(imgsize, imgsize, 3))
DGReal, DGFake = modelDG(imageReal), modelDG(imageFake)
combDG = Model(inputs=[imageReal, imageFake], outputs=[DGReal, DGFake])
combDG.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')

imageReal = Input(shape=(imgsize, imgsize, 3))
imageFake = Input(shape=(imgsize, imgsize, 3))
DFReal, DFFake = modelDF(imageReal), modelDF(imageFake)
combDF = Model(inputs=[imageReal, imageFake], outputs=[DFReal, DFFake])
combDF.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')

imageA = Input(shape=(imgsize, imgsize, 3))
imageB = Input(shape=(imgsize, imgsize, 3))
modelDG.trainable = False
modelDF.trainable = False
fakeB, fakeA = modelG(imageA), modelF(imageB)
disG, disF = modelDG(fakeB), modelDF(fakeA)
cycGF, cycFG = modelF(fakeB), modelG(fakeA)
combM = Model(inputs=[imageA, imageB], outputs=[disG, disF, cycGF, cycFG])
combM.compile(optimizer=Adam(adam_lr, adam_beta_1), loss=['mse', 'mse', 'mae', 'mae'],
              loss_weights=[1, 1, p_lambda, p_lambda])


def ImgGenerator(imgdir):
    imglst = [os.path.join(imgdir, x) for x in os.listdir(imgdir)]
    while True:
        random.shuffle(imglst)
        for fn in imglst:
            img = image.load_img(fn, target_size=(imgsize, imgsize))
            img = (image.img_to_array(img) - 127.5) / 127.5
            yield np.expand_dims(img, axis=0)


# if the memory is enough for all imgs ...
def ImgGeneratorS(imgdir):
    imglst = [os.path.join(imgdir, x) for x in os.listdir(imgdir)]
    X = np.zeros((len(imglst), imgsize, imgsize, 3))
    for i, fn in enumerate(imglst):
        if i % 20 == 0: print('%d/%d' % (i, len(imglst)))
        img = image.load_img(fn, target_size=(imgsize, imgsize))
        X[i] = image.img_to_array(img)
    X = (X - 127.5) / 127.5
    ids = list(range(len(imglst)))
    while True:
        random.shuffle(ids)
        for ii in ids: yield X[ii:ii + 1]


genA = ImgGeneratorS(imgdirA)
genB = ImgGeneratorS(imgdirB)
testA = ImgGenerator(testimgdirA)
testB = ImgGenerator(testimgdirB)

nb_batches = len(os.listdir(imgdirA)) // batch_size
ones = np.ones((batch_size, imgsize // 16, imgsize // 16, 1))
zeros = np.zeros((batch_size, imgsize // 16, imgsize // 16, 1))

recordG, recordF = [], []
for epoch in range(nb_epochs):
    print('Epoch %d of %d' % (epoch + 1, nb_epochs))
    progress_bar = Progbar(target=nb_batches)

    lossDG, lossDF = 0.5, 0.5
    for index in range(nb_batches):
        A_image_batch, B_image_batch = next(genA), next(genB)

        generateG = modelG.predict_on_batch(A_image_batch)
        recordG.append(generateG)
        if len(recordG) > 100: recordG = recordG[-50:]
        lossDG = combDG.train_on_batch([B_image_batch, random.choice(recordG)], [ones, zeros])[0]

        generateF = modelF.predict_on_batch(B_image_batch)
        recordF.append(generateF)
        if len(recordF) > 100: recordF = recordF[-50:]
        lossDF = combDF.train_on_batch([A_image_batch, random.choice(recordF)], [ones, zeros])[0]

        for _ in range(1):
            A_image_batch, B_image_batch = next(genA), next(genB)
            _, lossG, lossF, losscycGF, losscycFG = combM.train_on_batch([A_image_batch, B_image_batch],
                                                                         [ones, ones, A_image_batch, B_image_batch])

        progress_bar.update(index + 1,
                            values=[('DG', lossDG), ('G', lossG), ('DF', lossDF), ('F', lossF), ('cycGF', losscycGF),
                                    ('cycFG', losscycFG)])

    print('Testing for epoch {}:'.format(epoch + 1))

    modelG.save_weights(os.path.join(modeldir, 'modelG.h5'), True)
    modelF.save_weights(os.path.join(modeldir, 'modelF.h5'), True)
    modelDG.save_weights(os.path.join(modeldir, 'modelDG.h5'), True)
    modelDF.save_weights(os.path.join(modeldir, 'modelDF.h5'), True)

    tA = np.concatenate([next(testA) for x in range(4)], axis=0)
    tB = np.concatenate([next(testB) for x in range(4)], axis=0)
    gG = modelG.predict(tA, batch_size=1)
    gF = modelF.predict(tB, batch_size=1)
    tA = tA.reshape(-1, imgsize, 3)
    tB = tB.reshape(-1, imgsize, 3)
    gG = gG.reshape(-1, imgsize, 3)
    gF = gF.reshape(-1, imgsize, 3)
    img = np.concatenate([tA, gG, tB, gF], axis=1)
    img = (img * 127.5 + 127.5).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(testimgdir, 'plot_epoch_{0:03d}_generated.png'.format(epoch)))
