import time
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from caffe_classes import class_names

# # 加载模型
# start = time.clock()
# model = load_model('102flowersmodel.h5')
# print('Warming up took {}s'.format(time.clock() - start))
#
# # 图片预处理
# path = 'static/test_img/test.jpg'
# img_height, img_width = 224, 224
# x = image.load_img(path=path, target_size=(img_height, img_width))
# x = image.img_to_array(x)
# x = x[None]
#
# # 预测
# start = time.clock()
# y = model.predict(x)
# print('Prediction took {}s'.format(time.clock() - start))
#
# # 置信度
# for i in np.argsort(y[0])[::-1][:5]:
#     print('{}:{:.2f}%'.format(i, y[0][i] * 100))

def test_image():
    # 加载模型
    path = 'static/test_img/test.jpg'
    start = time.clock()
    model = load_model('102flowersmodel.h5')
    print('Warming up took {}s'.format(time.clock() - start))

    img_height, img_width = 224, 224
    x = image.load_img(path=path, target_size=(img_height, img_width))
    x = image.img_to_array(x)
    x = x[None]

    # 预测
    start = time.clock()
    y = model.predict(x)
    print('Prediction took {}s'.format(time.clock() - start))

    for i in np.argsort(y[0])[::-1][:5]:
        flag = i
        break
    dir=['0', '1', '10', '100', '101', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81',
          '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
    k=int(dir[flag])

    flower_class = class_names[k]
    return flower_class
# ppath='C:/Users/Administrator/Desktop/image_00009.jpg'
# print(test_image(ppath))