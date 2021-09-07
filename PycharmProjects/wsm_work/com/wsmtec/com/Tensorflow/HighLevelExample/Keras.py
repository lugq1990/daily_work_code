# -*- coding:utf-8 -*-
"""
    This is a class for a subclass of the Keras.Model, it is just an example
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.random.random((1000, 100))
label = np.random.randint(10, size=(1000,))
label = keras.utils.to_categorical(label, num_classes=10)

xtrain, xtest, ytrain, ytest = train_test_split(data, label, test_size=.2, random_state=1234)

# This is to implement the subclass
class myModel(keras.Model):

    def __init__(self, num_classes=10):
        super(myModel, self).__init__(name='myModel')
        self.num_classes = num_classes
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(.5)
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        return self.dense2(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

# This is the subclass of Keras.layer
class myLayer(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(myLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name='kernel', initializer='uniform', trainable=True, shape=shape)
        super(myLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(myLayer, self).get_config()
        base_config['output_dim'] = self.output_dim

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = keras.Sequential([myLayer(10), keras.layers.Activation('softmax')])
model.compile(loss=keras.losses.categorical_crossentropy, metrics=['accuracy'], optimizer='adam')
model.fit(xtrain, ytrain,epochs=10, batch_size=128, validation_data=(xtest, ytest))

print('Done!')
print('Eval accuracy:', model.evaluate(xtest, ytest, batch_size=128)[1])

# model = myModel(num_classes=10)
# model.compile(loss=keras.losses.categorical_crossentropy, metrics=['accuracy'], optimizer='adam')
#
# his = model.fit(xtrain, ytrain, epochs=100, batch_size=128, validation_data=(xtest, ytest))
# plt.plot(his.history['acc'], label='train')
# plt.plot(his.history['val_acc'], label='test')
# plt.legend()
# plt.show()
