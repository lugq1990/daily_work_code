# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import datetime
import time
import os
from Accenture.DeepLearning.Character_CNN_classification import helper, CharacterCNN
from tensorflow.contrib import learn

# Here use TensorFlow flags to get model training parameters
tf.flags.DEFINE_string('dev_sample_ratio', '.1', 'how many data for validation')
tf.flags.DEFINE_string('pos_file_path', "C:/Users/guangqiiang.lu/Documents/lugq/github/NLP"
                                        "/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.pos", 'pos_path')
tf.flags.DEFINE_string('neg_file_path', "C:/Users/guangqiiang.lu/Documents/lugq/github/NLP"
                                        "/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.neg", 'neg_path')
tf.flags.DEFINE_string('summery_path', "C:/Users/guangqiiang.lu/Documents/lugq/github/NLP/cnn-text-classification-tf-master/summaries",
                       'which path to write summary')

# Here is model parameters, such as embedding size, filter size, number filters .etc
tf.flags.DEFINE_integer('embedding_size', 128, 'embedding size')
tf.flags.DEFINE_string('filter_size', '3,4,5', 'filter size')
tf.flags.DEFINE_integer('num_filters', 128, 'how many filters')
tf.flags.DEFINE_float('drop_ratio', .5, 'drop out')
tf.flags.DEFINE_float('l2_reg', .0, 'l2 loss')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.flags.DEFINE_integer('num_epochs', 200, 'how many epochs for model training')
tf.flags.DEFINE_integer('evaluate_step', 50, 'how many epochs for evaluation')
tf.flags.DEFINE_integer('check_point_step', 50, 'how many epochs for checkpoint')
tf.flags.DEFINE_integer('num_checkpoints', 5, 'how many checkpoint to be stored')

# Some other training paramters
tf.flags.DEFINE_bool('allow_soft', True, 'Can be placed for soft place?')
tf.flags.DEFINE_bool('allow_log', False, 'whether or not to log the operations')


FLAGS = tf.flags.FLAGS

def process():
    print('Now is loading data')
    data, label = helper.load_data(FLAGS.pos_file_path, FLAGS.neg_file_path)

    max_document_lenght = max([len(x.split(' ')) for x in data])
    vocab = learn.preprocessing.VocabularyProcessor(max_document_lenght)
    x = np.array(list(vocab.transform(data)))

    # split data for training and testing
    from sklearn.model_selection import train_test_split
    np.random.seed(1234)
    xtrain, xtest, ytrain, ytest = train_test_split(x, label, test_size=.1, random_state=1234)

    print('Vocab size: {}'.format(len(vocab.vocabulary_)))
    return xtrain, xtest, ytrain, ytest, vocab

# This is main training function
def train(xtrain, xtest, ytrain, ytest, vocab):
    print('Start training process!')

    with tf.Graph().as_default():
        session_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft, log_device_placement=FLAGS.allow_log)
        sess = tf.Session(config=session_config)
        with sess.as_default():
            cnn = CharacterCNN.CharacterCNN(sequence_length=xtrain.shape[1],
                                            num_classes=ytrain.shape[1],
                                            vocab_size=len(vocab.vocabulary_),
                                            embedding_size=FLAGS.embedding_size,
                                            filter_size=list(map(int, FLAGS.filter_size.split(','))),
                                            num_filters=FLAGS.num_filters,
                                            l2_reg=FLAGS.l2_reg)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.losses)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track for gradient values and sparsity of variables
            grads_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram('{}/grad/his'.format(v.name), g)
                    sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                    grads_summaries.append(grad_hist_summary)
                    grads_summaries.append(sparsity_summary)

            grad_summmary_merge = tf.summary.merge(grads_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(FLAGS.summary_path, 'runs', timestamp))
            print('writing files to %s'%(out_dir))

            # Summary loss and accuracy
            loss_summary = tf.summary.scalar('loss', cnn.losses)
            acc_summary = tf.summary.scalar('acc', cnn.accuracy)

            # Training summary
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summmary_merge])
            train_summary_dir = os.path.join(FLAGS.summary_path, 'summary', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summary
            val_summary_op = tf.summary.merge([loss_summary, acc_summary])
            val_summary_dir = os.path.join(FLAGS.summary_path, 'summary', 'val')
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Here I also want to write checkpoint to directory
            checkpoint_dir = os.path.abspath(os.path.join(FLAGS.summary_path, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'checkpoints')
            if not os.path.exists(checkpoint_prefix):
                os.makedirs(checkpoint_prefix)

            # Here is a Saver object to save model
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab.save(os.path.join(FLAGS.summary_path, 'vocab'))

            # Start to train model
            sess.run(tf.global_variables_initializer())

            # First to get data from generator
            batchs = helper.batch_iter(list(zip(xtrain, ytrain)), FLAGS.batch_size, FLAGS.num_epochs)

            for batch in batchs:
                xbatch, ybatch = zip(*batch)
                feed_dict = {cnn.input_x: xbatch, cnn.input_y: ybatch, cnn.drop_ratio: FLAGS.drop_ratio}
                _, step, summeries, loss, accuracy = sess.run([train_op, global_step, train_summary_op,
                                                               cnn.losses, cnn.accuracy], feed_dict=feed_dict)
                time_str = datetime.datetime.now.isoformat()
                print('{}: step {}, loss={}, accuracy:{}'.format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summeries)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_step == 0:
                    print('Now is for evaluating')
                    feed_dict = {cnn.input_x: xbatch, cnn.input_y: ybatch, cnn.drop_ratio:1.}
                    step, summeries, loss, accuracy = sess.run([global_step, val_summary_op, cnn.losses, cnn.accuracy],
                                                               feed_dict=feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print('evaluting time {}:, step {}, loss={}, acc={}'.format(time_str, loss, accuracy))

                # store checkpoint
                if current_step % FLAGS.checkpoint_step == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=global_step)
                    print('Saved model checkpoint to {}\n'.format(path))


# HERE IS MAIN FUNCTION
def main(argv=None):
    xtrain, xtest, ytrain, ytest, vocab = process()
    train(xtrain, xtest, ytrain, ytest, vocab)

if __name__ == '__main__':
    tf.app.run()
