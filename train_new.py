import tensorflow as tf
from model.fcn import fcnRegressor
from config_folder_guard import config_folder_guard
from gen_batches import gen_batches
from logger import my_logger as logger
import pandas as pd
from PIL import Image


def read_csv_file():


    DATASET_MEDIUM_DIR = 'dataset_medium.csv'
    Image.MAX_IMAGE_PIXELS = None

    imgs_dirs = []
    dataset_read_result = pd.read_csv(DATASET_MEDIUM_DIR)
    i = 0
    for each_img_dir, \
        each_landmarks_dir, \
        each_target_image, \
        each_target_landmarks, \
        each_status in zip(dataset_read_result['Source image'],
                           dataset_read_result['Source landmarks'],
                           dataset_read_result['Target image'],
                           dataset_read_result['Target landmarks'],
                           dataset_read_result['status']):
        each_img_dir = 'images/' + each_img_dir
        each_landmarks_dir = 'landmarks/' + each_landmarks_dir
        each_target_image = 'images/' + each_target_image
        each_target_landmarks = 'landmarks/' + each_target_landmarks

        dataset_read_result.set_value(index=i, col='Source image', value=each_img_dir)
        dataset_read_result.set_value(index=i, col='Source landmarks', value=each_landmarks_dir)
        dataset_read_result.set_value(index=i, col='Target image', value=each_target_image)
        dataset_read_result.set_value(index=i, col='Target landmarks', value=each_target_landmarks)

        imgs_dirs.append(each_img_dir)
        i = i + 1

    print(dataset_read_result['Source image'][1])
    print(imgs_dirs[1])
    return dataset_read_result


dataset_read_result = read_csv_file()

# the first 10
source_image_array = dataset_read_result['Source image'][:1]
target_image_array = dataset_read_result['Target image'][:1]



from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array


import numpy as np


def model():
    x_1 = conv2d(x, 'Conv1', 32, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
    x_2 = tf.nn.avg_pool(x_1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pooling1')
    x_3 = conv2d(x_2, 'Conv2', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
    x_4 = tf.nn.avg_pool(x_3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pooling2')
    x_5 = conv2d(x_4, 'Conv3', 128, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
    x_6 = conv2d_transpose(x_5, 'deconv1', 64, [10, 8, 8, 64], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
    x_7 = conv2d(x_6, 'Conv4', 64, 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
    x_8 = conv2d_transpose(x_7, 'deconv2', 32, [10, 8, 8, 32], 3, 2, 'SAME', True, tf.nn.relu, self._is_train)
    # todo: change batch_size for conv2d_transpose output_shape x_6,x_8
    # x_9,x_10,x_11 as regression layer, corresponding Reg1,Reg2 and Reg3 respectively
    x_9 = reg(x_8, 'Reg1', 2, 3, 1, 'SAME', self._is_train)
    x_10 = reg(x_7, 'Reg2', 2, 3, 1, 'SAME', self._is_train)
    x_11 = reg(x_5, 'Reg3', 2, 3, 1, 'SAME', self._is_train)
source_image_tensor = []
target_image_tensor = []



for each_source_image_dir, each_target_image_dir in zip(source_image_array, target_image_array):
    # keras
    each_source_image = load_img(each_source_image_dir)
    each_target_image = load_img(each_target_image_dir)

    # 图片转化成array类型,因flow()接收numpy数组为参数
    each_source_image_array_type = img_to_array(each_source_image,
                                                data_format="channels_last")
    each_target_image_array_type = img_to_array(each_target_image,
                                                data_format="channels_last")
    each_source_image_array_type.astype('float32')/255.
    each_target_image_array_type.astype('float32')/255.
    # tensor
    each_source_image_array_type = np.reshape(each_source_image_array_type, each_source_image_array_type.shape)
    each_target_image_array_type = np.reshape(each_target_image_array_type, each_target_image_array_type.shape)
    source_image_tensor.append(each_source_image_array_type)
    target_image_tensor.append(each_target_image_array_type)


input_queue = tf.train.slice_input_producer([source_image_tensor, target_image_tensor], shuffle=True)
batch_x, batch_y = tf.train.batch(input_queue, batch_size=1)



