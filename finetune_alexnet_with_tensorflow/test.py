# -*- coding: utf-8 -*-

import tensorflow as tf
from alexnet import AlexNet  # import训练好的网络
import matplotlib.pyplot as plt
from caffe_classes import class_names

class_name = class_names  # oxford102种花的标签


def test_image(path_image, num_class, weights_path='Default'):
    # 把新图片进行转换
    img_string = tf.read_file(path_image)
    # img_decoded = tf.image.decode_png(img_string, channels=3)
    img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_resized = tf.reshape(img_resized, shape=[1, 227, 227, 3])

    # 图片通过AlexNet
    model = AlexNet(img_resized, 0.5, 102, skip_layer='', weights_path=weights_path)
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,
                      "/finetune_alexnet/model_epoch50.ckpt")  # 导入训练好的参数
        # score = model.fc8
        print(sess.run(model.fc8))
        prob = sess.run(max)[0]

        # 在matplotlib中观测分类结果
        plt.imshow(img_decoded.eval())
        plt.title("Class:" + class_name[prob])#到标签文件caffe_classes.py文件中根据prob查找对应的标签
        print(prob)
        plt.show()
    return class_name[prob]


#test_image('image_06785.jpg', num_class=102)  # 输入一张新图片