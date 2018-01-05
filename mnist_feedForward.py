#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import argparse
import sys
import tempfile
import numpy as np
import pandas as pd
import random
import tensorflow as tf


sample_size = 50000

# reading and splitting dataset by training samples and labels
print("Multilayer feed forward network. MNIST dataset. Sample size: " +
        str(sample_size))
pd_train = pd.read_csv('train.csv', header=None)
pd_test = pd.read_csv('test.csv', header=None)

#extract the labels of the samples
cols = [0]
label_train_temp = pd_train.as_matrix(pd_train.columns[cols])
label_test = pd_test.as_matrix(pd_test.columns[cols])

# drop the labels; only the training data remain
pd_train = pd_train.drop(pd_train.columns[cols], axis=1)
pd_test = pd_test.drop(pd_test.columns[cols], axis = 1)
data_train_temp = pd_train.as_matrix(columns = pd_train.columns)
data_test = pd_test.as_matrix(columns = pd_test.columns)

# shuffling the samples so that a different set of the given size is picked
# in each run. The number of samples will be chosen by the sample_size
randIDs = np.arange(0 , 50000)
np.random.shuffle(randIDs)
randIDs = randIDs[:sample_size]
data_train = [data_train_temp[ i] for i in randIDs]
label_train = [label_train_temp[ i] for i in randIDs]

# creates a one_hot vector for the given digit
def one_hot(i):
    a = np.zeros(10, 'uint8')
    a[i] = 1
    return a

# sends a set of random samples (stochastic batch)
def next_batch(num, data, labels):
    randIDs = np.arange(0 , len(data))
    np.random.shuffle(randIDs)
    randIDs = randIDs[:num]
    data_shuffle = [data[ i] for i in randIDs]
    labels_shuffle = [labels[ i] for i in randIDs]
    for i,j in enumerate(labels_shuffle):
        labels_shuffle[i] = one_hot(j)
    batch = []
    batch.append(np.asfarray(data_shuffle))
    batch.append(np.asfarray(labels_shuffle))
    return(batch)

# same as next_batch, except IDs are not suffled
def next_batch_test(num, data, labels):
    randIDs = np.arange(0 , len(data))
    randIDs = randIDs[:num]
    data_shuffle = [data[ i] for i in randIDs]
    labels_shuffle = [labels[ i] for i in randIDs]
    for i,j in enumerate(labels_shuffle):
        labels_shuffle[i] = one_hot(j)
    batch = []
    batch.append(np.asfarray(data_shuffle))
    batch.append(np.asfarray(labels_shuffle))
    return(batch)

def initWeights(shape):
    return(tf.Variable(tf.truncated_normal(shape, stddev=0.1)))
def initBiases(shape):
    return(tf.Variable(tf.constant(0.1, shape=shape)))


def main(_):
    # Import data

    # Create the model
    X = tf.placeholder(tf.float32, [None, 784])


    # 4 layer
    W0 = initWeights([784, 1024])
    b0 = initBiases([1024])
    y0 = tf.matmul(X, W0) + b0
    W1 = initWeights([1024, 64])
    b1 = initBiases([64])
    y1 = tf.matmul(y0, W1) + b1
    W2 = initWeights([64, 32])
    b2 = initBiases([32])
    y2 = tf.matmul(y1, W2) + b2
    W3 = initWeights([32, 10])
    b3 = initBiases([10])
    y =  tf.matmul(y2, W3) + b3


    '''
    # 3 layer
    W0 = initWeights([784, 64])
    b0 = initBiases([64])
    y0 = tf.matmul(X, W0) + b0
    W1 = initWeights([64, 32])
    b1 = initBiases([32])
    y1 = tf.matmul(y0, W1) + b1
    W2 = initWeights([32, 10])
    b2 = initBiases([10])
    y = tf.matmul(y1, W2) + b2
    '''


    '''
    # 2 layer
    W0 = initWeights([784, 32])
    b0 = initBiases([32])
    y0 = tf.matmul(X, W0) + b0
    W1 = initWeights([32, 10])
    b1 = initBiases([10])
    y = tf.matmul(y0, W1) + b1
    '''


    '''
    # 1 layer
    W0 = initWeights([784, 10])
    b0 = initBiases([10])
    y =  tf.matmul(X, W0) + b0
    '''


    # Define loss and optimizer
    y_actual = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y))
    training = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(30000):
        batch_xs, batch_ys = next_batch(50, data_train, label_train)
        sess.run(training, feed_dict={X: batch_xs, y_actual: batch_ys})

    # Test trained model
    batch_test = next_batch_test(10000, data_test, label_test)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={X: batch_test[0],
                                      y_actual: batch_test[1]}))
if __name__ == '__main__':
  tf.app.run(main=main)
