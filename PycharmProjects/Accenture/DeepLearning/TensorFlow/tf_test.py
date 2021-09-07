# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris

iris = load_iris()
x, y = iris.data, iris.target
y = keras.utils.to_categorical(y, num_classes=3)

model = Sequential()
model.add(Dense(32, input_dim=4))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

print(model.summary())

model.fit(x, y, epochs=10)