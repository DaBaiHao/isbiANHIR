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

def train():
    config = config_folder_guard({
        # train_parameters
        'image_size': [128, 128],
        'batch_size': 10,
        'learning_rate': 1e-4,
        'epoch_num': 500,
        'save_interval': 2,
        'shuffle_batch': True,
        # trainNet data folder
        'checkpoint_dir': 'training data/running data/checkpoints',
        'temp_dir': 'training data/running data/validate',
        'log_dir': 'training data/running data/log'
    })

    # 定义验证集和训练集

    dataset_read_result = read_csv_file()
    batch_x, batch_y = gen_batches(dataset_read_result[:400], {
        'batch_size': config['batch_size'],
        'image_size': config['image_size'],
        'shuffle_batch': config['shuffle_batch']
    })

    valid_x, valid_y = gen_batches(dataset_read_result[400:], {
        'batch_size': config['batch_size'],
        'image_size': config['image_size'],
        'shuffle_batch': config['shuffle_batch']
    })

    # import pdb
    # pdb.set_trace()
    # 断点

    # config['train_iter_num'] = len(os.listdir(train_x_dir)) // config["batch_size"]
    # config['valid_iter_num'] = len(os.listdir(valid_x_dir)) // config['batch_size']
    config['train_iter_num'] = 200
    config['valid_iter_num'] = 20

    #定义日志记录器
    train_log = logger(config['log_dir'], 'train.log')
    valid_log = logger(config['log_dir'], 'valid.log')

    #构建网络
    sess = tf.Session()
    reg = fcnRegressor(sess, True, config)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #开始训练
    print('start training')
    for epoch in range(config['epoch_num']):
        _train_L = []
        _train_L1 = []
        _train_L2 = []
        _train_L3 = []
        for i in range(config['train_iter_num']):
            _bx, _by = sess.run([batch_x, batch_y])
            _loss_train = reg.fit(_bx, _by)
            _train_L.append(_loss_train[0])
            _train_L1.append(_loss_train[1])
            _train_L2.append(_loss_train[2])
            _train_L3.append(_loss_train[3])
            print('[TRAIN] epoch={:>3d}, iter={:>5d}, loss={:.4f}, loss_1={:.4f}, loss_2={:.4f}, loss_3={:.4f}, loss_4={:.4f}, loss_5={:.4f}, loss_6={:.4f}'
                  .format(epoch + 1, i + 1, _loss_train[0], _loss_train[1], _loss_train[2], _loss_train[3], _loss_train[4], _loss_train[5], _loss_train[6]))
        print('[TRAIN] epoch={:>3d}, loss={:.4f}..................'.format(epoch + 1, sum(_train_L) / len(_train_L)))
        train_log.info('[TRAIN] epoch={:>3d}, loss={:.4f}, loss_1 = {:.4f}, loss_2 = {:.4f}, loss_3 = {:.4f}'
                       .format(epoch + 1, sum(_train_L) / len(_train_L), sum(_train_L1) / len(_train_L1), sum(_train_L2) / len(_train_L2), sum(_train_L3) / len(_train_L3)))

        #放入验证集进行验证
        _valid_L = []
        _valid_L1 = []
        _valid_L2 = []
        _valid_L3 = []
        for j in range(config['valid_iter_num']):
            _valid_x, _valid_y = sess.run([valid_x, valid_y])
            _loss_valid = reg.deploy(None, _valid_x, _valid_y)
            _valid_L.append(_loss_valid[0])
            _valid_L1.append(_loss_valid[1])
            _valid_L2.append(_loss_valid[2])
            _valid_L3.append(_loss_valid[3])
            print('[VALID] epoch={:>3d}, iter={:>5d}, loss={:.4f}, loss_1={:.4f}, loss_2={:.4f}, loss_3={:.4f}, loss_4={:.4f}, loss_5={:.4f}, loss_6={:.4f}'
                  .format(epoch + 1, j + 1, _loss_valid[0], _loss_valid[1], _loss_valid[2], _loss_valid[3], _loss_valid[4], _loss_valid[5], _loss_valid[6]))
        print('[VALID] epoch={:>3d}, loss={:.4f}..................'.format(epoch + 1, sum(_valid_L) / len(_valid_L)))
        valid_log.info('[VALID] epoch={:>3d}, loss={:.4f}, loss_1 = {:.4f}, loss_2 = {:.4f}, loss_3 = {:.4f}'
                       .format(epoch + 1, sum(_valid_L) / len(_valid_L), sum(_valid_L1) / len(_valid_L1), sum(_valid_L2) / len(_valid_L2), sum(_valid_L3) / len(_valid_L3)))

        if(epoch + 1) % config['save_interval'] == 0:
            _valid_x, _valid_y = sess.run([valid_x, valid_y])
            reg.deploy(config['temp_dir'], _valid_x, _valid_y)
            reg.save(sess, config['checkpoint_dir'])

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    train()