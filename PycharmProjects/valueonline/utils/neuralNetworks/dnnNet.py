# -*- coding:utf-8 -*-
import keras
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.models import Model
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import  train_test_split

class dnnNet(object):
    def __init__(self, n_classes=2, n_dims=None, n_layers=3, n_units=64, use_dropout=True, drop_ratio=.5, use_batchnorm=True,
                 metrics='accuracy', optimizer='rmsprop'):
        self.n_classes = n_classes
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.n_units = n_units
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_batchnorm = use_batchnorm
        self.metrics = metrics
        self.optimizer = optimizer
        self.model = self._init_model()

    def _init_model(self):
        if self.n_dims is None:
            raise AttributeError('Data Dimension must be provided!')
        inputs = Input(shape=(self.n_dims, ))

        # this is dense block function.
        def _dense_block(layers):
            res = Dense(self.n_units)(layers)
            if self.use_batchnorm:
                res = BatchNormalization()(res)
            res = Activation('relu')(res)
            if self.use_dropout:
                res = Dropout(self.drop_ratio)(res)
            return res

        for i in range(self.n_layers):
            if i == 0:
                res = _dense_block(inputs)
            else: res = _dense_block(res)

        if self.n_classes == 2:
            out = Dense(self.n_classes, activation='sigmoid')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        elif self.n_classes > 2:
            out = Dense(self.n_classes, activation='softmax')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        else:
            raise AttributeError('parameters n_class must be provide up or equal 2!')

        return model

    # For fit function, auto randomly split the data to be train and validation datasets.
    def fit(self, data, label, epochs=100, batch_size=256):
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
        self.his = self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1,
                                  validation_data=(xvalidate, yvalidate))
        print('Model evaluation on validation datasets accuracy:{:.4f}'.format(self.model.evaluate(xvalidate, yvalidate)[1]))
        return self

    def evaluate(self, data, label, batch_size=None, silent=False):
        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.6f}'.format(acc))
        return acc

    def predict(self, data, batch_size=None):
        return self.model.predict(data, batch_size=batch_size)

    def plot_acc_curve(self):
        style.use('ggplot')

        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.plot(self.his.history['acc'], label='Train Accuracy')
        ax1.plot(self.his.history['val_acc'], label='Validation Accuracy')
        ax1.set_title('Train and Validation Accuracy Curve')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy score')
        plt.legend()

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.plot(self.his.history['loss'], label='Train Loss')
        ax2.plot(self.his.history['val_loss'], label='Validation Loss')
        ax2.set_title('Train and Validation Loss Curve')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss score')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    x, y = iris.data, iris.target
    y = keras.utils.to_categorical(y, num_classes=3)

    model = dnnNet(n_classes=3, n_dims=4)
    model.fit(x, y, epochs=100)
    model.plot_acc_curve()