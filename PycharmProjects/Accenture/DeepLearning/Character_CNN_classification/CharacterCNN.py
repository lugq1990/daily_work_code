# -*- coding:utf-8 -*-
import tensorflow as tf

class CharacterCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_size, num_filters, l2_reg=0.):
        # First define input data, target data and dropout placeholder
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.drop_ratio = tf.placeholder(tf.float32, name='drop_ratio')

        # Here also with l2 loss
        l2_loss = tf.constant(0.)

        # First is embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.w = tf.get_variable('w', initializer=tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            self.embedded_char = tf.nn.embedding_lookup(self.w, self.input_x)
            self.embedded_char_expand = tf.expand_dims(self.embedded_char, -1)

        # According to different filter size to get pooling features result
        # This filter size is a list for different filters
        pooled_result = []
        for i, fil in enumerate(filter_size):
            with tf.name_scope('max_pool_%s'%(str(fil))):
                # Convolutinal size
                fil_shape = [fil, embedding_size, 1, num_filters]
                with tf.variable_scope('max_pool_%s'%(str(fil)), reuse=tf.AUTO_REUSE):
                    w = tf.get_variable('w', initializer=tf.truncated_normal(fil_shape, stddev=.1))
                    b = tf.get_variable('b', initializer=tf.truncated_normal([num_filters], stddev=.1))
                    conv = tf.nn.conv2d(self.embedded_char_expand, w, strides=[1,1,1,1], padding='VALID', name='conv')
                    relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    pool = tf.nn.max_pool(relu, ksize=[1, sequence_length-filter_size+1, 1, 1], strides=[1,1,1,1], padding=
                                          'VALID', name='pool')
                    pooled_result.append(pool)

        # Now We have get all max-pooling result, we have to concatenate them to be one
        n_filter_total = num_filters * len(filter_size)
        self.h_pool = tf.concat(pooled_result, 3)
        # Concatenate all max-pooling result, -1 means batch_size
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, n_filter_total])

        with tf.name_scope('drop'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.drop_ratio)

        # Get final result
        with tf.name_scope('output'):
            with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
                w = tf.get_variable('w', shape=[n_filter_total, num_classes],
                                    initializer=tf.random_normal_initializer)
                b = tf.get_variable('b', shape=[num_classes], initializer=tf.constant(.1))
                l2_loss += tf.nn.l2_loss(w)
                l2_loss += tf.nn.l2_loss(b)

                self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name='score')
                self.pred = tf.argmax(self.scores, 1, name='pred')

        # Crossentropy loss
        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.losses = loss + l2_loss * l2_reg

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.pred, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


