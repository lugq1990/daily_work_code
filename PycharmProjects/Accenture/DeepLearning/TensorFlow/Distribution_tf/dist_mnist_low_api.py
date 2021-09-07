# -*- coding:utf-8 -*-
import sys
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import math

# Get the training data from local file
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

FLAGS = None

image_size = 28
hidden_units = 100
hidden_units_2 = 128
batch_size = 32
training_step = 100
num_epochs = 10

ps_host = 'localhost:1001'
worker_host_1 = 'localhost:1101'
worker_host_2 = 'localhost:1102'

checkpoint_dir = "C:/Users/guangqiiang.lu/Documents/lugq/workings/201811/dist_tensorflow/checkpoint"

# task_index = int(sys.argv[1])
# # Is this uses as parameter server? True or False
# is_ps = sys.argv[2]
# # Is used as 'ps' OR 'worker'?
# job_name = str(sys.argv[3])
#
# # start to build distributed server and clusterSpec
# cluster = tf.train.ClusterSpec({'ps': ps_host, 'worker':[worker_host_1, worker_host_2]})
# server = tf.train.Server(cluster, task_index=task_index, job_name=job_name)

def dist_training(args):
    # Because of I make server outside of this function, it can't run.
    # So here I make server inside
    job_name = args['job_name']
    task_index = args['task_index']
    is_ps = args['is_ps']

    # start to build distributed server and clusterSpec
    cluster = tf.train.ClusterSpec({'ps': [ps_host], 'worker': [worker_host_1, worker_host_2]})
    print('*' * 10)
    print(job_name)
    server = tf.train.Server(cluster, task_index=task_index, job_name=job_name)

    if is_ps:
        server.join()
    else:
        is_chief = (task_index == 0)
        # tf.app.run(main=main, argv=[sys.argv[0]])
        # dist_training(server, is_chief)

        # Because of my model not training, whether is because of without device place? Test it.
        # For 'tf.train.replica_device_setter' is used for between-graph replication. Here just
        # place the in-graph replication.
        
        # with tf.device(
        #         tf.train.replica_device_setter(worker_device='/job:worker/task/%d'% task_index,
        #                                        cluster=cluster)):
        with tf.Device('/job:worker/task/%d'% task_index, cluster=cluster):

            print('&'*30)
            print('Now is in training step!')
            # First Define input data and label placeholder
            x = tf.placeholder(tf.float32, [None, 784])
            y = tf.placeholder(tf.float32, [None, 10])

            # Here is weight variables for 2 layers Dense layers
            with tf.name_scope('hidden1'):
                weight = tf.Variable(tf.random_normal([784, hidden_units], stddev=1./math(float(784))), name='weight')
                bias = tf.Variable(tf.random_normal([hidden_units]), name='bias')
                hidden1 = tf.nn.relu(tf.matmul(x, weight) + bias)

            # Second layer
            with tf.name_scope('hidden2'):
                weight = tf.Variable(tf.random_normal([hidden_units, hidden_units_2],
                                                      stddev=1./math(float(hidden_units))), name='weight')
                bias = tf.Variable(tf.random_normal([hidden_units_2]), name='bias')
                hidden2 = tf.nn.relu(tf.matmul(hidden1, weight)+ bias)

            # Here is output
            with tf.name_scope('out'):
                weight = tf.Variable(tf.random_normal([hidden_units_2, 10], stddev=1./math(float(10))), name='weight')
                bias = tf.Variable(tf.random_normal([10]), name='bias')
                logits = tf.matmul(hidden2, weight) + bias

            # Here is optimizer, loss and training optimizer
            global_step = tf.train.get_global_step()   # No checkpoint
            optimizer = tf.train.GradientDescentOptimizer(.1)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
            train_op = optimizer.minimize(loss, global_step=global_step)

            print('*'*100)
            print('Here is start training process:')
            ## Here is distributed monitor session
        with tf.train.MonitoredSession(master=server, checkpoint_dir=checkpoint_dir, is_chief=is_chief) as sess:
            while not sess.should_stop():
                for epoch in range(num_epochs):
                    batchx, batchy = mnist.train.next_batch(batch_size)

                    # Start training
                    _, l, g_step = sess.run([train_op, loss, global_step], feed_dict={x:batchx, y:batchy})
                    g_step += 1    # Add 1 to global step variable
                    if epoch % 1 == 0 :
                        print('Now is epoch %d, global_step :%d, loss:%.4f'%(epoch, g_step, l))

        print('Here I have finished model training step!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, required=True, help='Job name for distributing (ps or worker)')
    parser.add_argument('--task_index', type=int, required=True, help='Which index for training task')
    parser.add_argument('--is_ps', type=bool, required=True, help='Is ps or worker? Boolean type')

    # FLAGS, unparsed = parser.parse_known_args()
    # jobname = str(FLAGS.job_name)
    # task_index = int(FLAGS.task_index)
    # is_ps = str(FLAGS.is_ps)
    args = vars(parser.parse_args())
    dist_training(args)

    # job_name = args['job_name']
    # task_index = args['task_index']
    # is_ps = args['is_ps']

    # # start to build distributed server and clusterSpec
    # cluster = tf.train.ClusterSpec({'ps': [ps_host], 'worker': [worker_host_1, worker_host_2]})
    # print('*' * 10)
    # print(job_name)
    # server = tf.train.Server(cluster, task_index=task_index, job_name=job_name)
    #
    # if is_ps:
    #     server.join()
    # else:
    #     is_chief = (task_index == 0)
    #     # tf.app.run(main=main, argv=[sys.argv[0]])
    #     dist_training(server, is_chief)