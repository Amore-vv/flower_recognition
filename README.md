# flower_recognition
花卉识别程序


### finetune_alexnet_with_tensorflow(基于AlexNet的花卉识别)

finetune.py  训练神经网络的文件

alexnet.py AlexNet卷积神经网络模型

bvlc_alexnet.npy 训练好的参数

datagenerator.py 图片预处理

main.py 路由文件

运行时需要先训练神经网络，因为虽然写了检查点但是看不到检查点文件，训练好以后再运行main.py文件。训练时将oxford文件夹放在train.txt文件中的指定位置，或者根据oxford文件夹的位置对train.txt进行相应修改。

### vgg16_oxford_flower_102(基于VGG16迁移模型的花卉识别)

network.py 神经网络的初次训练

retrain.py 神经网络的再次训练

test.py 测试文件

102flowermodel.h5 训练好的模型保存

main.py 路由文件

url.txt 花卉百度百科链接

### 使用方法
运行main.py，访问http://127.0.0.1:5000/， 进入花卉识别主页面，点击“选择文件”按钮，上传要识别的花卉图片，然后点击“开始识别”按钮，进行识别。识别完成后，在页面下方显示识别图像、识别结果以及花卉相关介绍。

### 数据集下载地址

http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

### 训练好的参数文件 bvlc_alexnet.npy下载地址

http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
