import tensorflow as tf

# 1. 定义一个变量，必须给定初始值(图的构建，没有运行)
a = tf.Variable(tf.random_uniform([10000, 650], -0.05, 0.05))
b = tf.Dimension(a)
# 3. 进行初始化操作（推荐：使用全局所有变量初始化API）
# 相当于在图中加入一个初始化全局变量的操作
init_op = tf.global_variables_initializer()
print(type(init_op))

# 3. 图的运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 运行init op进行变量初始化，一定要放到所有运行操作之前
    sess.run(init_op)
    # init_op.run() # 这行代码也是初始化运行操作，但是要求明确给定当前代码块对应的默认session(tf.get_default_session())是哪个，底层使用默认session来运行
    # 获取操作的结果
    print("result:{}".format(sess.run(b)))