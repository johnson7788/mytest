# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

print(batch_size,n_batch)

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

w = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.matmul(x,w) + b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))