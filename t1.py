import tensorflow as tf

matrix1 = tf.constant(tf.zeros([784,10]))
matrix2 = tf.constant(tf.ones([1000,784]))
# matrix multiply, like np.dot(matrisx1,matrix2)
matrix_product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result1 = sess.run(matrix_product)
print(result1)