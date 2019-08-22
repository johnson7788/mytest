# import numpy as np
# def one_hot_encode(x):
#     """
#     One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
#     : x: List of sample Labels
#     : return: Numpy array of one-hot encoded labels
#     """
#     # TODO: Implement Function
#     targets = np.array(x).reshape(-1)
#     one_hot_targets = np.eye(10)[targets]
#     return one_hot_targets
#
#
# print(one_hot_encode(6))
# print(np.array(6).reshape(-1))
#
# x = [3,2,1,6,7,4,4,8]
# print(one_hot_encode(x))
#
# import tensorflow as tf
#
# def tensortest():
#     t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print(sess.run(tf.shape(t)))
#
# tensortest()


# import tensorflow as tf
# k = tf.constant([
#     [1, 0, 1],
#     [2, 1, 0],
#     [0, 0, 1]
# ], dtype=tf.float32, name='k')
# i = tf.constant([
#     [4, 3, 1, 0],
#     [2, 1, 0, 1],
#     [1, 2, 4, 1],
#     [3, 1, 0, 2]
# ], dtype=tf.float32, name='i')
# kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')
# image  = tf.reshape(i, [1, 4, 4, 1], name='image')
#
# res = tf.squeeze(tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "VALID"))
# # VALID means no padding
# with tf.Session() as sess:
#    print(sess.run(res))


from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier(max_iter=5)
print(clf.fit(X_features, y))
print(clf.score(X_features, y))