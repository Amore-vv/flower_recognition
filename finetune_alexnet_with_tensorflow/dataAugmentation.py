from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
    rotation_range=60)

img = load_img('image_00095.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x,
    batch_size=1,
    save_to_dir='D:output',   #保存在这个文件夹下
    save_prefix='lena',
    save_format='jpg'):
    i += 1
    if i > 20:  #生成20张图
        break
