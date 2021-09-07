# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

epochs = 2
batch_size = 32

path = "C:/Users/guangqiiang.lu/Documents/lugq/Kaggle/fashinemnist"

train = pd.read_csv(path + '/fashion-mnist_train.csv')
test = pd.read_csv(path + '/fashion-mnist_test.csv')


# Here is a data and label convert function
def get_features(df):
    data = df.iloc[:, 1:].values/255
    label = df.iloc[:, 0].values
    return data, label

xtrain, ytrain = get_features(train)
xtest, ytest = get_features(test)

# Because no matter keras model or tensorflow models, label needed to be one-hot
ytrain_1hot = tf.keras.utils.to_categorical(ytrain)
ytest_1hot = tf.keras.utils.to_categorical(ytest)


# Here start to build the model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dropout(.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

print('Model Summary:')
model.summary()

# Start to train model
his = model.fit(xtrain, ytrain_1hot, epochs=epochs, batch_size=batch_size, shuffle=True)

# After training, evaluate the model
print('Keras model accuracy:', model.evaluate(xtest, ytest_1hot)[1])

# After training model, I convert the keras model to tensorflow estimator
tf_estimator = tf.keras.estimator.model_to_estimator(keras_model=model)

input_name = model.input_names[0]
# Here is estimator training input_fn and test input_fn
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={input_name: xtrain},
                                                    y=ytrain, num_epochs=epochs,
                                                    batch_size=batch_size, shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x={input_name:xtest}, y=ytest, num_epochs=1, shuffle=False)

# Start to train the estimator model
tf_estimator.train(train_input_fn)

print('Estimator model evaluated accuracy:', tf_estimator.evaluate(test_input_fn))


# After all training process finised, store the model to disk
# Here is keras model, .h5 file
model.save(path + '/keras.h5')

# For tensorflow estimator
features_spec = {input_name: tf.FixedLenFeature(shape=[784], dtype=np.float32)}
serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(features_spec)
export_dir = tf_estimator.export_savemodel(export_dir_base=path, serving_input_receiver_fn=serving_fn)

print('All step finishe!!!')