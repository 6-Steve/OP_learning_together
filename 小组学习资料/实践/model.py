"""
    model.py: CNN神经网络模型
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def cnn_inference(images, batch_size, n_classes):
    """
        输入：
            images：队列中取的一批图片, 具体为：4D tensor [batch_size, width, height, 3]
            batch_size：每个批次的大小
            n_classes：n分类（这里是二分类，猫或狗）
        返回：
            softmax_linear：表示图片列表中的每张图片分别是猫或狗的预测概率（即：神经网络计算得到的输出值）。
                            例如: [[0.459, 0.541], ..., [0.892, 0.108]],
                            一个数值代表属于猫的概率，一个数值代表属于狗的概率，两者的和为1。
    """

    # 第一层的卷积层conv1，卷积核(weights)的大小是 3*3, 输入的channel(管道数/深度)为3, 共有16个
    with tf.variable_scope('conv1') as scope:

        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))   # 初始化为常数，通常偏置项biases就是用它初始化的


        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)      # 加入偏差，biases向量与矩阵的每一行进行相加, shape不变
        conv1 = tf.nn.relu(pre_activation, name='conv1')   # 在conv1的命名空间里，用relu激活函数非线性化处理

    # 第一层的池化层pool1和规范化norm1(特征缩放）
    with tf.variable_scope('pooling1_lrn') as scope:
        # 对conv1池化得到feature map
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        # lrn()：局部响应归一化, 一种防止过拟合的方法, 增强了模型的泛化能力，
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

        # 第二层的卷积层cov2，卷积核(weights)的大小是 3*3, 输入的channel(管道数/深度)为16, 共有16个
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],  # 这里的第三位数字16需要等于上一层的tensor维度
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # 第二层的池化层pool2和规范化norm2(特征缩放）
    with tf.variable_scope('pooling2_lrn') as scope:
        # 这里选择了先规范化再池化
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')

    # 第三层为全连接层local3
    # 连接所有的特征, 将输出值给分类器 (将特征映射到样本标记空间), 该层映射出256个输出
    with tf.variable_scope('local3') as scope:
        # 将pool2张量铺平, 再把维度调整成shape(shape里的-1, 程序运行时会自动计算填充)
        reshape = tf.reshape(pool2, shape=[batch_size, -1])

        dim = reshape.get_shape()[1].value            # 获取reshape后的列数
        weights = tf.get_variable('weights',
                                  shape=[dim, 256],   # 连接256个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 矩阵相乘再加上biases，用relu激活函数非线性化处理
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='local3')

    # 第四层为全连接层local4
    # 连接所有的特征, 将输出值给分类器 (将特征映射到样本标记空间), 该层映射出512个输出
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[256, 512],  # 再连接512个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 矩阵相乘再加上biases，用relu激活函数非线性化处理
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # 第五层为输出层(回归层): softmax_linear
    # 将前面的全连接层的输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',
                                  shape=[512, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    # 这里没做归一化和交叉熵。真正的softmax函数放在下面的losses()里面和交叉熵结合在一起了，这样可以提高运算速度。
    # 图片列表中的每张图片分别被每个分类取到的概率，
    return softmax_linear


def losses(logits, labels):
    """
        输入：
            logits: 经过cnn_inference得到的神经网络输出值（图片列表中每张图片分别是猫或狗的预测概率）
            labels: 图片对应的标签（即：真实值。用于与logits预测值进行对比得到loss）
        返回：
            loss： 损失值（label真实值与神经网络输出预测值之间的误差）
    """
    with tf.variable_scope('loss') as scope:
        # label与神经网络输出层的输出结果做对比，得到损失值（这做了归一化和交叉熵处理）
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='loss_per_eg')
        loss = tf.reduce_mean(cross_entropy, name='loss')  # 求得batch的平均loss（每批有16张图）
    return loss


def training(loss, learning_rate):
    """
        输入：
            loss: 训练中得到的损失值
            learning_rate：学习率
        返回：
            train_op: 训练的最优值。训练op，这个参数要输入sess.run中让模型去训练。
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        global_step = tf.Variable(0, name='global_step', trainable=False)  # 全局步数赋值为0

        train_op = optimizer.minimize(loss, global_step=global_step)   # 以最大限度地最小化loss

    return train_op


def evaluation(logits, labels):
    """
        输入：
            logits: 经过cnn_inference得到的神经网络输出值（图片列表中每张图片分别是猫或狗的预测概率）
            labels: 图片对应的标签（真实值，0或1）
        返回：
            accuracy：准确率（当前step的平均准确率。即：这些batch中多少张图片被正确分类了）
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)        # 转换格式为浮点数
        accuracy = tf.reduce_mean(correct)            # 计算当前批的平均准确率
    return accuracy
