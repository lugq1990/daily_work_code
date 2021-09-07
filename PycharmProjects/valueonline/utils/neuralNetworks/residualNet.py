# -*- coding:utf-8 -*-
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Dense, Dropout, Activation, GlobalAveragePooling1D, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

class residualNet(object):
    def __init__(self, input_dim1=None, input_dim2=None, n_classes=2, n_layers=4, flatten=True, use_dense=True,
                 n_dense_layers=1, conv_units=64, stride=1, padding='SAME', dense_units=128, drop_ratio=.5,
                 optimizer='rmsprop', metrics='accuracy'):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.flatten = flatten
        self.use_dense = use_dense
        self.n_dense_layers = n_dense_layers
        self.conv_units = conv_units
        self.stride = stride
        self.padding = padding
        self.dense_units = dense_units
        self.drop_ratio = drop_ratio
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = self._init_model()

    def _init_model(self):
        inputs = Input(shape=(self.input_dim1, self.input_dim2))

        # dense net residual block
        def _res_block(layers):
            res = Conv1D(self.conv_units, self.stride, padding=self.padding)(layers)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            res = Conv1D(self.input_dim2, self.stride, padding=self.padding)(res)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            return keras.layers.add([layers, res])

        # construct residual block chain.
        for i in range(self.n_layers):
            if i == 0:
                res = _res_block(inputs)
            else:
                res = _res_block(res)

        # using flatten or global average pooling to process Convolution result
        if self.flatten:
            res = Flatten()(res)
        else:
            res = GlobalAveragePooling1D()(res)

        # whether or not use dense net, also with how many layers to use
        if self.use_dense:
            for j in range(self.n_dense_layers):
                res = Dense(self.dense_units)(res)
                res = BatchNormalization()(res)
                res = Activation('relu')(res)
                res = Dropout(self.drop_ratio)(res)

        if self.n_classes == 2:
            out = Dense(self.n_classes, activation='sigmoid')(res)
            model = Model(inputs, out)
            print('Model structure:')
            model.summary()
            model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        elif self.n_classes > 2:
            out = Dense(self.n_classes, activation='softmax')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        else:
            raise AttributeError('parameters n_classes must up to 2!')

        return model

    # Fit on given training data and label. Here I will auto random split the data to train and validation data,
    # for test datasets, I will just use it if model already trained then I will evaluate the model.
    def fit(self, data, label, epochs=100, batch_size=256):
        # label is not encoding as one-hot, use keras util to convert it to one-hot
        if len(label.shape) == 1:
            label = keras.utils.to_categorical(label, num_classes=len(np.unique(label)))

        xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
        self.his = self.model.fit(xtrain, ytrain, verbose=1, epochs=epochs,
                                  validation_data=(xvalidate, yvalidate), batch_size=batch_size)
        print('After training, model accuracy on validation datasets is {:.2f}%'.format(
            self.model.evaluate(xvalidate, yvalidate)[1]*100))
        return self

    # this is evaluation function to evaluate already trained model.
    def evaluate(self, data, label, batch_size=None, silent=False):
        if len(label.shape) == 1:
            label = keras.utils.to_categorical(label, num_classes=len(np.unique(label)))

        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.2f}%'.format(acc*100))
        return acc

    def predict(self, data, batch_size=None):
        return self.model.predict(data, batch_size=batch_size)

    # plot after training accuracy and loss curve.
    def plot_acc_curve(self):
        style.use('ggplot')

        fig1, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['acc'], label='Train Accuracy')
        ax.plot(self.his.history['val_acc'], label='Validation Accuracy')
        ax.set_title('Train and Validation Accruacy Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy score')
        plt.legend()

        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['loss'], label='Traing Loss')
        ax.plot(self.his.history['val_loss'], label='Validation Loss')
        ax.set_title('Train and Validation Loss Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss score')

        plt.legend()
        plt.show()


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    x, y = iris.data, iris.target
    x = x.reshape(-1, 2, 2)
    #y = keras.utils.to_categorical(y, num_classes=3)

    model = residualNet(n_classes=3, input_dim1=2, input_dim2=2)
    model.fit(x, y)
    model.evaluate(x, y)
    # model.plot_acc_curve()

