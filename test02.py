import numpy as np
import tensorflow as tf
# a=np.array([[1,2,3], [4,5,6]]).shape
# print(a)
#
#
# tensor_1d = np.array([1.3, 1, 4.0, 23.99])
# print(tensor_1d)
# print(tensor_1d[0])
#
# tensor_2d = np.array([(1,2,3,4),(4,5,6,7),(8,9,10,11),(12,13,14,15)])
# print(len(tensor_2d.shape))
# print(tensor_2d)
# print(tensor_2d[3][2])

# matrix1 = np.array([(2,2,2),(2,2,2),(2,2,2)], dtype='int32')
# matrix2 = np.array([(1,1,1),(1,1,1),(1,1,1)], dtype='int32')
#
# print(matrix1)
# print(matrix2)
#
# matrix1 = tf.constant(matrix1)
# matrix2 = tf.constant(matrix2)
# matrix_product = tf.matmul(matrix1, matrix2)
# matrix_sum = tf.add(matrix1, matrix2)
# matrix_3 = np.array([(2,7,2), (1,4,2), (9,0,2)], dtype='float32')
# print(matrix_3)
#
# matrix_det = tf.matrix_determinant(matrix_3)
# with tf.Session() as sess:
#     result1 = sess.run(matrix_product)
#     result2 = sess.run(matrix_sum)
#     result3 = sess.run(matrix_det)
#
# print(result1)
# print(result2)
# print(result3)
#
#
# print(np.random.rand(100, 1))
# X = 2 * np.random.rand(100,1)
#
# print(np.ones((100,1)))
# print(np.c_[np.ones((100,1)),X])
# np.c_

# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# print(a.dot(b))


# from numpy.linalg import inv
# a = np.array([[1., 2.], [3., 4.]])
# print(inv(a))
#
# a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
# print(inv(a))

# np.random.seed(28)
# N = 100
# x = np.linspace(0, 6, N) + np.random.normal(loc=0.0, scale=2, size=N)
# y = 14 * x - 7 + np.random.normal(loc=0.0, scale=5.0, size=N)
# print(x)
# print(y)
#
# print(np.random.normal(loc=0.0, scale=2, size=N))

# print(s)
# #Verify the mean and the variance:
# print(np.mean(s))
# print(abs(mu - np.mean(s)) < 1)
# print(np.std(s, ddof=1))
# #verify variance
# print(abs(sigma - np.std(s, ddof=1)) < 1)


# mu, sigma = 0, 1 # mean and standard deviation
# s = np.random.normal(mu, sigma, 1000)
#
# import matplotlib.pyplot as plt
# count, bins, ignored = plt.hist(s, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
#          linewidth=2, color='r')
# plt.show()

# a = [1,2,3,4,5]
# print(np.mean(a))
# print(np.median(a))
# print(np.sqrt(
#     np.mean(
#         np.abs(a - np.mean(a))**2)))
# print(np.std(a))

import pandas as pd

# d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
# #    'Age':pd.Series([25,26,25,23,30,29,23]),
# #    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
# #
# # df = pd.DataFrame(d)
# # print(df['Age']>=30)

# df = pd.DataFrame([[4, 9],] * 3, columns=['A', 'B'])
# print(df)
# print(df.apply(np.sqrt))
# print(df.apply(np.sum, axis=0))
# print(df.apply(lambda x: x>5))
# a = [1,2,3,4,5]
# b = [2,3,4,5,6]
# print(a)
# print(b)
# print(np.dot(a,b))


# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 2, 3, 4, 5, 6])
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(X, Y)
# GaussianNB(priors=None, var_smoothing=1e-09)
# print(clf.predict([[-1.6, -1]]))
#
# clf_pf = GaussianNB()
# clf_pf.partial_fit(X, Y, np.unique(Y))
# GaussianNB(priors=None, var_smoothing=1e-09)
# print(clf_pf.predict([[-0.8, -1]]))

# import numpy as np
# logits = [2.0, 1.0, 0.1]
# exps = [np.exp(i) for i in logits]
# print(exps)
# sum_of_exps = sum(exps)
# softmax = [j/sum_of_exps for j in exps]
# print(softmax)
# print(sum(softmax))

# import tensorflow as tf
# #
# # x = tf.constant([[2, 2], [3, 3]])
# # y = tf.constant([[8, 16], [2, 3]])
# # power = tf.pow(x, y)  # [[256, 65536], [9, 27]]
# #
# # with tf.Session() as sess:
# #     pw = sess.run([power])
# #
# # print(pw)

# import math
# # Pslowi = 2/4 = 0.5
# # Pfasti = 2/4  =0.5
# # entropy = -Pslowi * log2(Pslowi) + (-Pfasti * log2(Pfasti) )
# print(-0.5*math.log2(0.5)-0.5*math.log2(0.5))


# # -*- coding: utf-8 -*-
# import tensorflow as tf
#
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(w1)
#     print(sess.run(w1))


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
