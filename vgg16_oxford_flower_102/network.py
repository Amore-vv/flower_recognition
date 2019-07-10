import os
import glob
import math
import numpy as np
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
# 数据集
train_dir = 'data/train'  # 训练集
validation_dir = 'data/valid'  # 验证集
nb_epoch = 50  # 迭代次数，原项目默认1000次
batch_size = 32  # 批量大小
img_size = (224, 224)  # 图片大小
freeze_layers_number = 0  # 冻结层数

classes = sorted([o for o in os.listdir(train_dir)])  # 根据文件名分类
nb_train_samples = len(glob.glob(train_dir + '/*/*.*'))  # 训练样本数
nb_validation_samples = len(glob.glob(validation_dir + '/*/*.*'))  # 验证样本数

# 定义模型
base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=img_size + (3,)),
                                classes=len(classes))  # 预训练的VGG16网络，替换掉顶部网络

for layer in base_model.layers:  # 保留原有网络全部参数
    print(layer.trainable)
    layer.trainable = False

x = base_model.output  # 自定义网络
x = Flatten()(x)  # 展平
x = Dense(4096, activation='elu', name='fc1')(x)  # 全连接层，激活函数elu
x = Dropout(0.6)(x)  # Droupout 0.6
x = Dense(4096, activation='elu', name='fc2')(x)
x = Dropout(0.6)(x)
predictions = Dense(len(classes), activation='softmax', name='predictions')(x)  # 输出层，指定类数

model = Model(input=base_model.input, output=predictions)  # 新网络=预训练网络+自定义网络

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), metrics=['accuracy'])
print(model.summary())

train_datagen = ImageDataGenerator(rotation_range=30., shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True)  # 30°内随机旋转，0.2几率应用错切，0.2几率缩放内部，水平随机旋转一半图像
train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))  # 去掉imagenet BGR均值
train_data = train_datagen.flow_from_directory(train_dir, target_size=img_size, classes=classes)
validation_datagen = ImageDataGenerator()  # 用于验证，无需数据增强
validation_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))
validation_data = validation_datagen.flow_from_directory(validation_dir, target_size=img_size,
                                                         classes=classes)


# 训练&保存
def get_class_weight(d):
    '''
    calculate the weight of each class
    :param d: dir path
    :return: a dict
    '''
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    class_number = dict()
    dirs = sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))])
    k = 0
    for class_name in dirs:
        class_number[k] = 0
        iglob_iter = glob.iglob(os.path.join(d, class_name, '*.*'))
        for i in iglob_iter:
            _, ext = os.path.splitext(i)
            if ext[1:] in white_list_formats:
                class_number[k] += 1
        k += 1
    total = np.sum(list(class_number.values()))
    max_samples = np.max(list(class_number.values()))
    mu = 1. / (total / float(max_samples))
    keys = class_number.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu * total / float(class_number[key]))
        class_weight[key] = score if score > 1. else 1.

    return class_weight


class_weight = get_class_weight(train_dir)  # 计算每个类别所占数据集的比重

early_stopping = EarlyStopping(verbose=1, patience=30, monitor='val_loss')  # 30次微调后loss仍没下降便迭代下一轮
model_checkpoint = ModelCheckpoint(filepath='102flowersmodel.h5', verbose=1, save_best_only=True, monitor='val_loss')
callbacks = [early_stopping, model_checkpoint]

model.fit_generator(train_data, steps_per_epoch=nb_train_samples / float(batch_size), epochs=nb_epoch,
                    validation_data=validation_data, validation_steps=nb_validation_samples / float(batch_size),
                    callbacks=callbacks, class_weight=class_weight)

print('Training is finished!')