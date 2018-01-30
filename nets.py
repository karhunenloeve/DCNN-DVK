#!/usr/bin/python

"""
Copyright 2017 Luciano Melodia

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import scipy.io as sio
import os
import config as cfg
import datetime
import visualize
import math

from theano import tensor as T
from keras.models import *
from keras.layers import Conv2D, AveragePooling2D, UpSampling2D, LocallyConnected2D, ZeroPadding2D, Cropping2D, ConvLSTM2D, LSTM, MaxPooling2D, SeparableConv2D
from keras.layers import UpSampling3D, MaxPooling3D, Conv3D, LeakyReLU, GaussianNoise
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten, Reshape, Input, AlphaDropout, GaussianDropout
from keras.layers.merge import concatenate
from keras.optimizers import Nadam
from keras import losses
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, TensorBoard
from scipy import stats
from numpy import mean, std
from keras import applications
from keras.models import load_model
from keras.backend import minimum, maximum
from keras.applications.vgg19 import VGG19
from keras import regularizers
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt

def IoU_metric(y_true, y_pred, smooth=K.epsilon()):
    min = K.sum(minimum(K.abs(y_true), K.abs(y_pred)))
    max = K.sum(maximum(K.abs(y_true), K.abs(y_pred)))
    sum = (min + smooth) / (max + smooth)
    return K.mean(sum)


def IoU(y_true, y_pred):
    return 1 - IoU_metric(y_true, y_pred)


def mean_squared_error(y_true, y_pred):
    return losses.mse(y_true, y_pred)

def misc(y_true, y_pred):
    return 0.5 * mean_squared_error(y_true, y_pred) + 0.5 * IoU(y_true, y_pred)

class uNet:
    trainQuantil = 0
    batch_size = 0
    epochs = 0

    def __init__(self):
        self.trainQuantil = cfg.settings['trainQuantil']
        self.batch_size = cfg.settings['batch_size']
        self.epochs = cfg.settings['epochs']

    def load(self, path, name, appendix="", save=False, boost=False, fac=10, amount=10, norm=False):
        """
        :param path: .mat files required for reading with saved double array (no struct arrays here)
        :param name: name of the resulting file
        :param save: save file (true/false)
        :param mode: decide wether to use npz or direct access mode
        :param appendix: appendix will be added at the end of the filename as an extension
        :param boost: upsamling the data (true/false)
        :param fac: upsampling factor
        :param norm: normalization (true/false
        :return:
        Function returns an array of the required data.
        Data has to be 4-dim. Path specifies the path to the data folder.
        """
        files = sorted(os.listdir(path))
        target = os.path.realpath("data")
        data = []

        for f in files:
            amount = amount - 1
            if f.endswith(".mat"):
                dvk = sio.loadmat(path + f)
                data.append(dvk[name])
                if amount == 0:
                    break

        data = np.array(data)
        v, w, x, y, z = data.shape
        shaped = data.reshape(v * z, w, x, y)

        if norm == True:
            mean_data = np.mean(shaped)
            std_data = np.std(shaped)
            shaped = (shaped - mean_data) / std_data

        if boost == True:
            for dim in range(len(data.shape) - 2):
                if dim == 0:
                    pass
                else:
                    shaped = np.repeat(shaped, fac, axis=dim)

        if save == True:
            np.savez_compressed(target + '/' + name + appendix, a = shaped)

        return(shaped)

    def store(self, path, name, appendix="", save=False, boost=False, fac=10, amount=10, norm=False):
        self.load(path, name, appendix, save, boost, fac, amount, norm)
        return "Data has been loaded."

    def restore(self, start_path, target_path):
        """
        :param start_path: npz file of compressed densities (sparse coded)
        :param target_path: npz file of compressed kernels(sparse coded)
        :return:
        """
        start = np.load(start_path)
        target = np.load(target_path)
        return start['a'], target['a']

    def plot_data(self, data, color="red", save=True, name="figure", datatype=".svg"):
        """
        :param data: one 3-Dim datapoint to be displayed in a mesh grid
        :param color: set the color of the mesh grid, either as string or RGB
        :param save: save as .svg image
        :param name: the name of the images, which has to be saved
        :param datatype: datatype of the images, which has to be saved
        :return:
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        z, x, y = data.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, -z, zdir='z', c=color)

        if save == True:
            plt.savefig(name + datatype)

    def get_layer_content(self, model):
        inp = model.input  # input placeholder
        outputs = [layer.output for layer in model.layers]  # all layer outputs
        functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

        test = np.random.random(input_shape)[np.newaxis, ...]
        layer_outs = [func([test, 1.]) for func in functors]
        return layer_outs

    def pca(self, trainingData):
        # performing PCA with the Data
        a, b, c, d = trainingData_shuffled.shape
        trainingData = trainingData.reshape(a, b * c * d)
        myPCA = PCA(trainingData)
        variance = myPCA.s / np.sum(myPCA.s)

        plt.plot(variance)
        plt.show()

        trainingData = myPCA.Y
        trainingData = trainingData.reshape(10000, 9, 9, 9)

        return trainingData

    def uNet(self, trainingData):
        """
        :param trainingData: numpy array of the training data.
        :return:
        """
        input_shape = (trainingData.shape)[1:4]
        inputs = Input(shape=input_shape)
        
        re1 = Reshape((27,27,1))(inputs)
        up1 = UpSampling2D((2,2))(re1)
        noise1 = GaussianNoise(1)(up1)

        conv1 = Conv2D(8, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l1_l2(0.005))(noise1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(8, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(16, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(16, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = AveragePooling2D(pool_size=2)(conv1)

        conv2 = Conv2D(32, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(pool1)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(32, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(32, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(32, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(32, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)

        up3 = UpSampling2D((6,6))(conv2)
        merge3 = concatenate([drop1, up3])
        conv3 = Conv2D(32, 4, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(merge3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(32, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(16, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(16, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(8, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(8, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(4, 3, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)

        conv3 = Conv2D(1, 1, activation="sigmoid")(conv3)
        outputs = Reshape((9,9,9))(conv3)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer = Nadam(lr=0.5*1e-4), loss = IoU, metrics = ["MAE", mean_squared_error, IoU_metric])

        return model

    def evaluate(self, trainingPath, targetPath, modelPath):
        trainingData, targetData = [], []
        dvk_train, dvk_goal = sio.loadmat(trainingPath), sio.loadmat(targetPath)
        trainingData.append(dvk_train["density_f"])
        targetData.append(dvk_goal["kernel_f"])
        trainingData, targetData = np.array(trainingData), np.array(targetData)
        v, w, x, y, z = trainingData.shape
        trainingData, targetData = trainingData.reshape(v * z, w, x, y), targetData.reshape(v * z, w, x, y)

        mean_trainingData, mean_targetData = np.mean(trainingData), np.mean(targetData)
        std_trainingData, std_targetData = np.std(trainingData), np.std(targetData)
        trainingData, targetData = (trainingData - mean_trainingData) / std_trainingData, (targetData - mean_targetData) / std_targetData

        model = self.uNet(trainingData)
        model.load_weights(modelPath)

        x = np.arange(trainingData.shape[0])
        np.random.shuffle(x)
        train = trainingData[x]
        test = targetData[x]

        print(model.evaluate(x=train, y=test))

    def train(self, trainingPath, targetPath, mode = "dir", name="simpleModel"):
        """
        :param trainingPath: path to the npz file of the training data or path to the directory of the training data
        :param targetPath: path to the npz file of the target data or path to the directory of the target data
        :param mode: decide wether to use npz or direct access mode
        :param appendix: appendix will be added at the end of the filename as an extension
        :param boost: upsamling the data (true/false)
        :param fac: upsampling factor
        :param amount: amount of files to read to build the dataset
        :param norm: normalization (true/false)
        :param map64: files are 9x9x9 -> 90x90x90, upsamling to 63x63x63 leads to an even representation of 64x64x64 by adding 0 as parameter
        :return:
        """
        if mode == "npz":
            trainingData, targetData = self.restore(trainingPath, targetPath)
        elif mode == "dir":
            trainingData = self.load(trainingPath, "density_f", appendix="", boost=False, fac=3, amount=1, norm=True)
            targetData = self.load(targetPath, "kernel_f", appendix="", boost=False, fac=3, amount=1, norm=True)

        x = np.arange(trainingData.shape[0])
        np.random.shuffle(x)
        trainingData_shuffled = trainingData[x]
        targetData_shuffled = targetData[x]

        # separating the data
        quantil_upper = len(trainingData) - (round(len(trainingData) * self.trainQuantil))
        x= trainingData_shuffled[0:quantil_upper]
        y = targetData_shuffled[0:quantil_upper]
        x_valid = trainingData_shuffled[quantil_upper:len(trainingData)]
        y_valid = targetData_shuffled[quantil_upper:len(targetData)]

        # training the model
        early_stopping = EarlyStopping(monitor='loss', min_delta=1e-4, patience=15, verbose=0, mode='auto')
        model_checkpoint = ModelCheckpoint('./model/checkpoint.hdf5', monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=10)
        csv_logger = CSVLogger('./model/model_1.log')
        tbcallback = TensorBoard(log_dir='./Graph', histogram_freq=1, write_grads=True, batch_size=self.batch_size, write_graph=False, write_images=False)

        model = self.uNet(trainingData)
        model_callbacks = [model_checkpoint, csv_logger, tbcallback, early_stopping]
        model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(x_valid, y_valid), shuffle=True, callbacks=model_callbacks)
        model.save('model/' + name + '.h5')

        print("Finished with the training.")
        return print("Done!")

    def predict(self, data, model):
        model = load_model(model)
        return(model.predict(data))


obj = uNet()
obj.train("data/density/", "data/kernel/")
