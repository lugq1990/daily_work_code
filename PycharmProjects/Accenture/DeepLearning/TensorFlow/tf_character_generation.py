# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time

# enable Eager execution for using optimizer with Tape
tf.enable_eager_execution()

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file).read()

# Get unique vocab
vocab = sorted(set(text))

# Process data for converting character to int index also converting index to character
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)   # This is the unique vocab

text_as_int = np.array([char2idx[c] for c in text])

# Show how this character with indexed
# for char, _ in zip(char2idx, range(20)):
#     print('{:6s} ---> {:4d}'.format(repr(char), char2idx[char]))
# print('{} ---> {}'.format(text[:10], text_as_int[:10]))


# Making the dataset to be for training and testing, for training data is 'hell', then target data is 'ello'
seq_length = 100
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length+1, drop_remainder=True)

# Print chunks
# for item in chunks.take(5):
#   print(repr(''.join(idx2char[item.numpy()])))


# Write train and target split function
def split_input_target(chunk):
    inp = chunk[:-1]
    tar = chunk[1:]
    return inp, tar

datasets = chunks.map(split_input_target)

# Print the first dataset, with different step input and target datasets(Here is 10 steps)
# for inp, tar in datasets.take(1):
#     print('Input data: ', repr(''.join(idx2char[inp.numpy()])))
#     print('Target Data: ', repr(''.join(idx2char[tar.numpy()])))
#     for i, (inp_idx, tar_idx) in enumerate(zip(inp[:10], tar[:10])):
#         print('Step {:4d}'.format(i))
#         print('input: {} ({:s})'.format(inp_idx, idx2char[inp_idx]))
#         print('target: {} ({:s})'.format(tar_idx, idx2char[tar_idx]))


# Set batchsize and shuffle size
batch_size = 100
buffer_size = 10000
datasets = datasets.shuffle(buffer_size).batch(batch_size)


# Here write a model class for inheriting the keras class
class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, units):
        super(Model, self).__init__()
        self.vocal_size = vocab_size
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)

        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                                return_sequences=True,
                                                recurrent_initializer='glorot_uniform',
                                                stateful=True)
        else:
            self.gru = tf.keras.layers.GRU(self.units,
                                           return_sequences=True,
                                           recurrent_initializer='glorot_uniform',
                                           recurrent_activation='sigmoid', stateful=True)

        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        embedding = self.embedding(x)
        out = self.gru(embedding)
        pred = self.fc(out)
        return pred

vocab_size = len(vocab)
embedding_size = 256
units = 1024

# Here is building the model
model = Model(vocab_size, embedding_size, units)

# Make the model optimizer and loss function
optimizer = tf.train.AdamOptimizer()
def loss_fun(real, pred):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=pred)

# Set checkpoint for model training
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


model.build(tf.TensorShape([batch_size, seq_length]))
print(model.summary())


# Here is the training step
epochs = 20
for epoch in range(epochs):
    start = time.time()

    # initialize the hidden state at every training start step
    hidden = model.reset_states()

    for (batch, (inp, target)) in enumerate(datasets):
        with tf.GradientTape() as tape:
            # get the model prediction
            pred = model(inp)
            loss = loss_fun(target, pred)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))
        if batch % 100 == 0:
            print('epoch {} batch {} loss {}'.format(epoch+1, batch, loss))

    if (epoch+1)% 5 == 0:
        # save the checkpoint
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} loss {:.4f}'.format(epoch+1, loss))
    print('Time for 1 epoch {} seconds'.format(time.time()- start))


# --- Here is prediction step:
# restore from the checkpoint
model = Model(vocab_size, embedding_size)

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

# Start to generate strings
num_generate = 1000

start_string = 'Q'

# convert the string to numbers
input_eval = [char2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)

# Generated result
text_generated = []
# How big the temperature
temperature = 1.

model.reset_states()
for i in range(num_generate):
    pred = model(input_eval)
    # remove batch_size dimension
    pred = tf.squeeze(pred, 0)

    # using the multinomial distribution to predict the word generated
    pred = pred / temperature
    pred_id = tf.multinomial(pred, num_samples=1)[-1, 0].numpy()

    # change the input_eval for next generation
    input_eval = tf.expand_dims([pred_id], 0)

    # get the model prediction
    text_generated.append(idx2char[pred_id])

print('Here is model prediction: ', text_generated)
