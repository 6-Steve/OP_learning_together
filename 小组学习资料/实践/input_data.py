"""
    input_data.py: 读取训练数据
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os


def get_files(file_dir):
    """
        输入：
            file_dir：存放训练图片的文件地址
        返回:
            image_list：乱序后的图片路径列表
            label_list：乱序后的标签(相对应图片)列表
    """
    # 建立空列表
    cats = []           # 存放是猫的图片路径地址
    label_cats = []     # 对应猫图片的标签
    dogs = []           # 存放是猫的图片路径地址
    label_dogs = []     # 对应狗图片的标签

    # 从file_dir路径下读取数据，存入空列表中
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])              # 所有行，列=0（选中所有猫狗图片路径地址），即重新存入乱序后的猫狗图片路径
    label_list = list(temp[:, 1])              # 所有行，列=1（选中所有猫狗图片对应的标签），即重新存入乱序后的对应标签
    label_list = [int(float(i)) for i in label_list]  # 把标签列表转化为int类型（用列表解析式迭代，相当于精简的for循环）

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
        输入：
            image,label：要生成batch的图像和标签
            image_W，image_H: 图像的宽度和高度
            batch_size: 每个batch（小批次）有多少张图片数据
            capacity: 队列的最大容量
        返回：
            image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
            label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    image = tf.cast(image, tf.string)   # 将列表转换成tf能够识别的格式
    label = tf.cast(label, tf.int32)


    input_queue = tf.train.slice_input_producer([image, label])   # 生成队列(牵扯到线程概念，便于batch训练), 将image和label传入

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)   # 使用JPEG的格式解码从而得到图像对应的三维矩阵。


    # 图片数据预处理：统一图片大小(缩小图片) + 标准化处理
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)                  # 将image转换成float32类型
    image = tf.image.per_image_standardization(image)   # 图片标准化处理，加速神经网络的训练

    # 按顺序读取队列中的数据
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    return image_batch, label_batch


