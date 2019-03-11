import tensorflow as tf
import numpy as np


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



"""
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
"""


def gen_batches(dataset_read_result):
    source_image_array = dataset_read_result['Source image']
    target_image_array = dataset_read_result['Target image']

    # 如果参考图像数量和待配准图像数量不同，那么意味着出错了
    assert len(source_image_array) == len(target_image_array)

    source_image_array = tf.cast(source_image_array, tf.string)
    target_image_array = tf.cast(target_image_array, tf.string)

    input_queue = tf.train.slice_input_producer([source_image_array, target_image_array])
    source_image_array = tf.read_file(input_queue[0])
    target_image_array = tf.read_file(input_queue[1])
    source_image_array = tf.image.decode_jpeg(source_image_array, channels=3)
    target_image_array = tf.image.decode_jpeg(target_image_array, channels=3)

    # resize
    source_image_array = tf.image.resize_images(source_image_array, [128, 128], method=tf.image.ResizeMethod.BICUBIC)
    target_image_array = tf.image.resize_images(target_image_array, [128, 128], method=tf.image.ResizeMethod.BICUBIC)

    source_image_array = tf.cast(source_image_array, tf.float32)
    target_image_array = tf.cast(target_image_array, tf.float32)

    source_image_array = tf.image.per_image_standardization(source_image_array)
    target_image_array = tf.image.per_image_standardization(target_image_array)

    # 标准化数据
    source_image_array, target_image_array = tf.train.batch([source_image_array, target_image_array],
                                                            batch_size=4,
                                                            num_threads=32,  # 线程
                                                            capacity=256)
    return source_image_array, target_image_array


source_image_array, target_image_array = gen_batches(dataset_read_result)


def inference(source_image_array):
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, source_image_array.get_shape()[-1], 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 initializer=tf.constant_initializer(0.))

        conv1 = tf.nn.conv2d(source_image_array, weights, strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, biases)
        conv1 = tf.nn.relu(conv1, name=scope.name)
    return conv1


batch_size = 2
img_height = 128
img_width = 128
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_height, img_width, 1])
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_height, img_width, 1])

#xy = tf.concat([x, y], axis=3)
# fcn_out = inference(xy)


# def losses(logits, labels):





sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in np.arange(10):

    train_logits2 = sess.run(train_logits)

