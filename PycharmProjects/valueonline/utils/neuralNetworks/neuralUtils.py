# -*- coding:utf-8 -*-
import numpy as np
import tensorflow.keras as keras

def check_label_shape(label):
    if len(label.shape) == 1:
        label = keras.utils.to_categorical(label, num_classes=len(np.unique(label)))

    return label