#!/usr/bin/python

"""
Copyright 2018 Luciano Melodia

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

import scipy.io as sio
import config as cfg
import visualize as vs
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import *
from keras.layers import Conv2D, Deconv2D, UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Reshape, Input
from keras.layers.merge import multiply
from keras.optimizers import Nadam
from keras import losses
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.backend import minimum, maximum
from sklearn.decomposition import PCA
from matplotlib.mlab import PCA as matpca


def IoU_metric(y_true, y_pred, smooth=K.epsilon()):
    min = K.sum(minimum(K.abs(y_true), K.abs(y_pred)))
    max = K.sum(maximum(K.abs(y_true), K.abs(y_pred)))
    sum = (min + smooth) / (max + smooth)
    return K.mean(sum)

def IoU(y_true, y_pred):
    return 1 - IoU_metric(y_true, y_pred)

def mean_squared_error(y_true, y_pred):
    return losses.mse(y_true, y_pred)

def clinic_loss(y_true, y_pred):
    loss = K.sum(K.tf.multiply(((y_true - y_pred)**2),(y_true/K.sum(y_true))))*100
    return loss

def absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred)

def misc(y_true, y_pred):
    return 0.5 * mean_squared_error(y_true, y_pred) + 0.5 * IoU(y_true, y_pred)

class uNet:
    trainQuantil = 0
    batch_size = 0
    epochs = 0
    PCA = False
    ndim = 9

    def __init__(self):
        self.trainQuantil = cfg.settings['trainQuantil']
        self.batch_size = cfg.settings['batch_size']
        self.epochs = cfg.settings['epochs']

    def load(self, path, name, appendix="", save=False, boost=False, fac=10, amount=30, norm=False):
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
        data = np.empty([9,9,9,0])

        for f in files:
            amount = amount - 1
            f=f.replace("._","")
            if f.endswith(".mat"):
                dvk = sio.loadmat(path + f)
                data = np.concatenate((data, dvk[name]), axis=3)
                if amount == 0:
                    break

        data = data.T

        if norm == True:
            data = (0.9-0.1) * (data - data.min(axis=(0,1,2,3), keepdims=False)) / (data.max(axis=(0,1,2,3), keepdims=False)-data.min(axis=(0,1,2,3), keepdims=False)) + 0.1

        if boost == True:
            for dim in range(len(data.shape) - 2):
                if dim == 0:
                    pass
                else:
                    data = np.repeat(data, fac, axis=dim)

        if save == True:
            np.savez_compressed(target + '/' + name + appendix, a = data)

        return(data)

    def store(self, path, name, appendix="", save=False, boost=False, fac=10, amount=30, norm=False):
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
        a, b, c, d = trainingData.shape
        trainingData = trainingData.reshape(a, b * c * d)
        myPCA = PCA(self.ndim)
        dataPCA = matpca(trainingData)

        plt.figure(1)
        plt.plot(dataPCA.fracs)
        plt.xlabel('Dimensionen', fontsize=12)
        plt.ylabel('Anteil an der Gesamtvarianz', fontsize=12)
        plt.savefig("model/test.svg")

        trainingData = myPCA.fit_transform(trainingData)
        trainingData = trainingData.reshape(10000, 3, 3, 1)

        return trainingData

    def uNet(self, trainingData):
        """
        :param trainingData: numpy array of the training data.
        :return:
        """
        input_shape = (9, 9, 9)
        inputs = Input(shape=input_shape)

        re1 = Reshape((27, 27, 1))(inputs)
        up1 = UpSampling2D((2, 2))(re1)

        conv1 = Conv2D(8, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l1(0.005))(up1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(8, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(16, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(16, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        drop1 = Dropout(0.2)(conv1)
        pool1 = AveragePooling2D(pool_size=2)(drop1)

        conv2 = Conv2D(32, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(pool1)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(32, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(32, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(32, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(32, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv2)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)

        up3 = UpSampling2D((6, 6))(conv2)
        merge3 = concatenate([drop1, up3])
        conv3 = Conv2D(32, 4, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(merge3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(32, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(16, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(16, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(8, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(8, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(4, 3, kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(0.001))(conv3)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)

        conv3 = Conv2D(1, 1, activation="sigmoid")(conv3)

        outputs = Reshape((9, 9, 9))(conv3)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        model.load_weights("model/checkpoint.hdf5")
        model.compile(optimizer = Nadam(lr=1e-5), loss = IoU, metrics = ["MAE", mean_squared_error, IoU_metric, clinic_loss])

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

        model.evaluate(x=train, y=test)

    def train(self, trainingPath, targetPath, mode = "dir", name="model"):
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
        Example
        object = uNet()
        object.train("data/density/", "data/kernel/")
        :return:
        """
        if mode == "npz":
            trainingData, targetData = self.restore(trainingPath, targetPath)
        elif mode == "dir":
            trainingPath=trainingPath.replace("._","")
            targetPath=targetPath.replace("._","")
            trainingData = self.load(trainingPath, "density_f", appendix="", boost=False, fac=3, amount=30, norm=True)
            targetData = self.load(targetPath, "kernel_f", appendix="", boost=False, fac=3, amount=30, norm=True)

        x = np.arange(trainingData.shape[0])
        np.random.shuffle(x)
        if self.PCA == False:
            trainingData_shuffled = trainingData[x]
        else:
            trainingData_shuffled = self.pca(trainingData[x])

        targetData_shuffled = targetData[x]

        # separating the data
        quantil_upper = len(trainingData) - (round(len(trainingData) * self.trainQuantil))
        x= trainingData_shuffled[0:quantil_upper]
        y = targetData_shuffled[0:quantil_upper]
        x_valid = trainingData_shuffled[quantil_upper:len(trainingData)]
        y_valid = targetData_shuffled[quantil_upper:len(targetData)]

        # training the model
        early_stopping = EarlyStopping(monitor='loss', min_delta=1e-6, patience=50, verbose=1, mode='auto')
        model_checkpoint = ModelCheckpoint('./model/checkpoint.hdf5', monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=10)
        csv_logger = CSVLogger('./model/model_1.log')
        tbcallback = TensorBoard(log_dir='./Graph', histogram_freq=1, write_grads=True, batch_size=self.batch_size, write_graph=False, write_images=False)

        model = self.uNet(trainingData_shuffled)
        model_callbacks = [model_checkpoint, csv_logger, early_stopping] # tbcallback,
        model.fit(x, y, validation_data=(x_valid, y_valid), batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True, callbacks=model_callbacks)
        model.save('model/' + name + '.h5')
        return "Done!"

    def predict(self, model, train, goal, organ="pizza", print_x = 33, print_y = 4, plot = True):
        """
        :param model: path to the saved keras model
        :param train: path to the .mat training file
        :return: goal: path to the .mat target file
        Example command:
        object = uNet()
        object.predict("model/simpleModel.h5", "data/density/dense_0.mat", "data/kernel/kernel_0.mat")
        """
        model = load_model(model, custom_objects={"IoU": IoU, "IoU_metric": IoU_metric, "clinic_loss": clinic_loss})

        trainingData, targetData = np.empty([9,9,9,0]), np.empty([9,9,9,0])

        dvk_train, dvk_goal = sio.loadmat(train), sio.loadmat(goal)
        trainingData = np.concatenate((trainingData, dvk_train["density_f"]), axis=3)
        targetData = np.concatenate((targetData, dvk_goal["kernel_f"]), axis=3)
        trainingData, targetData = np.array(trainingData), np.array(targetData)

        trainingData = np.array(trainingData).T
        targetData = np.array(targetData).T

        trainingData = (0.9-0.1) * (trainingData - trainingData.min(axis=(0,1,2,3), keepdims=False)) / (trainingData.max(axis=(0,1,2,3), keepdims=False)-trainingData.min(axis=(0,1,2,3), keepdims=False)) + 0.1
        targetData = (0.9-0.1) * (targetData - targetData.min(axis=(0,1,2,3), keepdims=False)) / (targetData.max(axis=(0,1,2,3), keepdims=False)-targetData.min(axis=(0,1,2,3), keepdims=False)) + 0.1

        datapoints = trainingData
        datatargetpoints = targetData
        x,y = print_x, print_y

        prediction = model.predict(datapoints)

        eingang = datapoints[x][y]
        ziel = datatargetpoints[x][y]
        vorhersage = prediction[x][y]
        fehler = np.sqrt((ziel - vorhersage) ** 2)

        if plot == True:
            data = [eingang, ziel, vorhersage, fehler]
            vs.show_field(data, organ=organ)
        return prediction
