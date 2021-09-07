# -*- coding:utf-8 -*-
"""This class is For LSTM model, also can use it for build bidirectional LSTM, use parameter use_bidire """
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation, LSTM, Bidirectional, GRU
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
#from .neuralUtils import check_label_shape

class lstmNet(object):
    def __init__(self, n_classes=2, input_dim1=None, input_dim2=None, n_layers=3, use_dropout=True, drop_ratio=.5,
                 use_bidirec=False, use_gru=False, rnn_units=64, use_dense=True, dense_units=64, use_batch=True,
                 metrics='accuracy', optimizer='rmsprop'):
        self.n_classes = n_classes
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_layers = n_layers
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_bidierc = use_bidirec
        self.use_gru = use_gru
        self.rnn_units = rnn_units
        self.use_dense = use_dense
        self.use_batch = use_batch
        self.dense_units = dense_units
        self.metrics = metrics
        self.optimizer = optimizer
        self.model = self._init_model()

    def _init_model(self):
        inputs = Input(shape=(self.input_dim1, self.input_dim2))

        def _lstm_block(layers, name_index=None):
            if self.use_bidierc:
                res = Bidirectional(LSTM(self.rnn_units, return_sequences=True,
                                         recurrent_dropout=self.drop_ratio), name='bidi_lstm_'+str(name_index))(layers)
            elif self.use_gru:
                res = GRU(self.rnn_units, return_sequences=True,
                          recurrent_dropout=self.drop_ratio, name='gru_'+str(name_index))(layers)
            else:
                res = LSTM(self.rnn_units, return_sequences=True,
                           recurrent_dropout=self.drop_ratio, name='lstm_'+str(name_index))(layers)

            if self.use_dropout:
                res = Dropout(self.drop_ratio)(res)

            return res

        # No matter for LSTM, GRU, bidirection LSTM, final layer can not use 'return_sequences' output.
        for i in range(self.n_layers - 1):
            if i == 0:
                res = _lstm_block(inputs, name_index=i)
            else:
                res = _lstm_block(res, name_index=i)

        # final LSTM layer
        if self.use_bidierc:
            res = Bidirectional(LSTM(self.rnn_units), name='bire_final')(res)
        elif self.use_gru:
            res = GRU(self.rnn_units, name='gru_final')(res)
        else:
            res = LSTM(self.rnn_units, name='lstm_final')(res)

        # whether or not to use Dense layer
        if self.use_dense:
            res = Dense(self.dense_units, name='dense_1')(res)
            if self.use_batch:
                res = BatchNormalization(name='batch_1')(res)
            res = Activation('relu')(res)
            if self.use_dropout:
                res = Dropout(self.drop_ratio)(res)

        if self.n_classes == 2:
            out = Dense(self.n_classes, activation='sigmoid', name='out')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        elif self.n_classes > 2:
            out = Dense(self.n_classes, activation='softmax', name='out')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        else:
            raise AttributeError('parameter n_class must be provide up or equals to 2!')

        return model

    def fit(self, data, label, epochs=100, batch_size=256):
        #label = check_label_shape(label)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
        self.his = self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1,
                                  validation_data=(xvalidate, yvalidate))
        print('Model evaluation on validation datasets accuracy:{:.2f}'.format(
            self.model.evaluate(xvalidate, yvalidate)[1]*100))
        return self

    def evaluate(self, data, label, batch_size=None, silent=False):
        #label = check_label_shape(label)

        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.2f}'.format(acc*100))
        return acc

    def predict(self, data, batch_size=None):
        return self.model.predict(data, batch_size=batch_size)

    def plot_acc_curve(self, plot_acc=True, plot_loss=True, figsize=(8, 6)):
        style.use('ggplot')

        if plot_acc:
            fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax1.plot(self.his.history['acc'], label='Train accuracy')
            ax1.plot(self.his.history['val_acc'], label='Validation accuracy')
            ax1.set_title('Train and validation accuracy curve')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy score')
            plt.legend()

        if plot_loss:
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
            ax2.plot(self.his.history['loss'], label='Train Loss')
            ax2.plot(self.his.history['val_loss'], label='Validation Loss')
            ax2.set_title('Train and validation loss curve')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss score')
            plt.legend()

        plt.show()


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    x, y = iris.data, iris.target
    x = x.reshape(-1, 2, 2)
    y = keras.utils.to_categorical(y, num_classes=3)

    model = lstmNet(n_classes=3, input_dim1=2, input_dim2=2, use_gru=True)
    model.fit(x, y, epochs=10)
    model.plot_acc_curve()