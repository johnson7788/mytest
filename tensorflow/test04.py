import numpy as np
# valid_score = [True]
# nonzero_numerator, nonzero_denominator = np.array([True]), np.array([True])
# numerator = np.array([1.5])
# denominator = np.array([29.1875])
# output_scores = np.ones(1)
# print(numerator[True])
# output_scores[valid_score] = 1 - (numerator[valid_score] /
#                                   denominator[valid_score])
# # arbitrary set to zero to avoid -inf scores, having a constant
# # y_true is not interesting for scoring a regression anyway
# print(output_scores[nonzero_numerator & nonzero_denominator])
# output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
# print(output_scores)
#
#
# a = np.reshape(np.arange(16), (4,4)) # create a 4x4 array of integers
# print(a)
# print(a[False])
# print(a[True])

# 2 by 2 matrices
# w1  = np.array([[1, 2], [3, 4]])
# w2  = np.array([[5, 6], [7, 8]])
#
# # flatten
# w1_flat = np.reshape(w1, -1)
# w2_flat = np.reshape(w2, -1)
#
# print(w1_flat)
# print(w2_flat)
# w = np.concatenate((w1_flat, w2_flat))
# array([1, 2, 3, 4, 5, 6, 7, 8])


# w0 = 2*np.random.random((3,4)) -1
# print(w0)

import tensorflow as tf

# n_features = 120
# n_labels = 5
# weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
# init_op = tf.global_variables_initializer()
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # 运行init op进行变量初始化，一定要放到所有运行操作之前
#     sess.run(init_op)
#     # init_op.run() # 这行代码也是初始化运行操作，但是要求明确给定当前代码块对应的默认session(tf.get_default_session())是哪个，底层>使用默认session来运行
#     # 获取操作的结果
#     print("result:{}".format(sess.run(weights)))


# def run():
#     output = None
#     logit_data = [1.2, 0.9, 0.4]
#     logits = tf.placeholder(tf.float32)
#     softmax = tf.nn.softmax(logits)
#     with tf.Session() as sess:
#         output = sess.run(softmax,feed_dict={logits:logit_data})
#     return output
#
#
# def cross_entropy():
#     softmax_data = [0.7, 0.2, 0.1]
#     one_hot_data = [1.0, 0.0, 0.0]
#
#     softmax = tf.placeholder(tf.float32)
#     one_hot = tf.placeholder(tf.float32)
#
#     cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
#
#     with tf.Session() as sess:
#         print(sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot:one_hot_data}))
#
#
# def steady():
#     a = 1000000000
#     for i in range(1000000):
#         a = a + 1e-6
#     print(a - 1000000000)
#
# import math

#
# from pprint import pprint
#
# # 4 Samples of features
# example_features = [
#     ['F11','F12','F13','F14'],
#     ['F21','F22','F23','F24'],
#     ['F31','F32','F33','F34'],
#     ['F41','F42','F43','F44']]
# # 4 Samples of labels
# example_labels = [
#     ['L11','L12'],
#     ['L21','L22'],
#     ['L31','L32'],
#     ['L41','L42']]
#
# # PPrint prints data structures like 2d arrays, so they are easier to read
# pprint(batches(3, example_features, example_labels))

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    output_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)

    return output_batches

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# TODO: Set batch size
batch_size = 128
assert batch_size is not None, 'You must set the batch size'

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # TODO: Train optimizer on all batches
    # for batch_features, batch_labels in ______
    for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
        sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))