"""
    test.py: 用训练好的模型对随机一张图片进行猫狗预测
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import matplotlib.pyplot as plt
import input_data
import model
import numpy as np
import cv2
import os
from model import evaluation

def get_one_image(img_list):
    """
        输入：
            img_list：图片路径列表
        返回：
            image：从图片路径列表中随机挑选的一张图片
    """
    n = len(img_list)                  # 获取文件夹下图片的总数
    ind = np.random.randint(0, n)      # 从 0~n 中随机选取下标
    img_dir = img_list[ind]            # 根据下标得到一张随机图片的路径

    image = Image.open(img_dir)        # 打开img_dir路径下的图片
    image = image.resize([208, 208])   # 改变图片的大小，定为宽高都为208像素
    image = np.array(image)            # 转成多维数组，向量的格式
    return image

def get_images(img_list):
    images = []  # 初始化一个空列表来存储加载的图片
    labels  = []  # 初始化一个空列表来存储图片的标签


    for img_path in os.listdir(img_list):
        # 使用OpenCV加载图片，注意这里默认图片是彩色的，使用cv2.IMREAD_COLOR
        img_path_ = os.path.join(img_list, img_path)
        img = cv2.imread(img_path_, cv2.IMREAD_COLOR)

        filename = os.path.basename(img_path_)
        label = filename.split('.')[0]
        # 可以根据需要添加图片预处理步骤，例如调整大小、归一化等
        img = cv2.resize(img, (208,208))  # 调整大小
        # img = img.astype('float32') / 255.0  # 归一化到[0, 1]范围

        images.append(img)
        if label == "cat":
            label = 0
        elif label == "dog":
            label = 1
        labels.append(label)
    # 将图片列表转换为NumPy数组，方便后续处理
    images = np.array(images)

    return images,labels  # 返回加载的图片数据

def evaluate_model():
    # 修改成自己测试集的文件夹路径
    test_dir = 'E:/classify/cats_vs_dogs-main/cats_vs_dogs/data/evaluate_model/'
    # 获取测试集的图片路径列表
    images_array , labels = get_images(test_dir)          # 从测试集中选取图片

    BATCH_SIZE = len(images_array)
    N_CLASSES = 2               # 还是二分类(猫或狗)

    image = tf.cast(images_array, tf.float32)                    # 将列表转换成tf能够识别的格式
    image = tf.image.per_image_standardization(image)           # 图片标准化处理
    image = tf.reshape(image, [len(images_array) , 208, 208, 3])                 # 改变图片的形状
    logit = model.cnn_inference(image, BATCH_SIZE, N_CLASSES)   # 得到神经网络输出层的预测结果
    logit = tf.nn.softmax(logit)                                # 进行归一化处理（使得预测概率之和为1）

    accuracy = evaluation(logit, labels)

    print('test accuracy = %.2f%%' % (accuracy* 100.0))

    # 修改成自己训练好的模型路径
    logs_train_dir = 'E:/classify/cats_vs_dogs-main/cats_vs_dogs/log/'

def evaluate_one_image():
    # 修改成自己测试集的文件夹路径
    test_dir = 'E:/classify/cats_vs_dogs-main/cats_vs_dogs/data/test/'
    # test_dir = '/home/user/Dataset/cats_vs_dogs/test/'

    test_img = input_data.get_files(test_dir)[0]   # 获取测试集的图片路径列表
    image_array = get_one_image(test_img)          # 从测试集中随机选取一张图片

    # 将这个图设置为默认图，会话设置成默认对话，这样在with语句外面也能使用这个会话执行。
    with tf.Graph().as_default():    # 参考：https://blog.csdn.net/nanhuaibeian/article/details/101862790
        BATCH_SIZE = 1               # 这里我们要输入的是一张图(预测这张随机图)
        N_CLASSES = 2                # 还是二分类(猫或狗)

        image = tf.cast(image_array, tf.float32)                    # 将列表转换成tf能够识别的格式
        image = tf.image.per_image_standardization(image)           # 图片标准化处理
        image = tf.reshape(image, [1, 208, 208, 3])                 # 改变图片的形状
        logit = model.cnn_inference(image, BATCH_SIZE, N_CLASSES)   # 得到神经网络输出层的预测结果
        logit = tf.nn.softmax(logit)                                # 进行归一化处理（使得预测概率之和为1）

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])         # x变量用于占位，输入的数据要满足这里定的shape

        # 修改成自己训练好的模型路径
        logs_train_dir = 'E:/classify/cats_vs_dogs-main/cats_vs_dogs/log/'

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("从指定路径中加载模型...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)   # 读取路径下的checkpoint
            if ckpt and ckpt.model_checkpoint_path:                # checkpoint存在且其存放的变量不为空
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('模型加载成功, 训练的步数为： %s' % global_step)
            else:
                print('模型加载失败，checkpoint文件没找到！')

            prediction = sess.run(logit, feed_dict={x: image_array})   # 得到预测结果
            max_index = np.argmax(prediction)                          # 获取预测结果的最大值的下标
            if max_index == 0:
                pre = prediction[:, 0][0] * 100
                print('图片是猫的概率为： {:.2f}%'.format(pre))       # 下标为0，则为猫，并打印是猫的概率
            else:
                pre = prediction[:, 1][0] * 100
                print('图片是狗的概率为： {:.2f}%'.format(pre))       # 下标为1，则为狗，并打印是狗的概率

    plt.imshow(image_array)                                        # 接受图片并处理
    plt.show()                                                     # 显示图片


if __name__ == '__main__':
    # 调用方法，开始测试
    evaluate_one_image()
    #evaluate_model()
