import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils.load_tfrecord import input_fn
from hparms import *

NUM_HIDDEN_NODES = 2500


def main():
    # image_mean = np.load("../utils/image_mean.npy")
    keras.backend.clear_session()
    sess = tf.Session()

    # data_dir = ['../../data/2019_08_07/', '../../data/2019_08_10/', '../../data/2019_08_12/', '../../data/2019_08_12_2/']
    data_dir = ['../../data/2019_08_07/']
    dataset = input_fn(is_training=True, data_dir=data_dir, batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, traj_batch = iterator.get_next()

    for i in range(2000):
        im, la = sess.run([image_batch, traj_batch])
        im = im * 255.0
        im = im.astype(np.uint8)

        image = cv2.cvtColor(im[0], cv2.COLOR_BGR2RGB)
        print(la)
        cv2.imshow('image', image)
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break
        print(i)
    sess.close()


# def main():
#     # Load the CIFAR10 data.
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
#     # Input image dimensions.
#     input_shape = x_train.shape[1:]
#
#     # Normalize data.
#     x_train = x_train.astype('float32') / 255
#     x_test = x_test.astype('float32') / 255
#
#     # If subtract pixel mean is enabled
#     x_train_mean = np.mean(x_train, axis=0)
#     x_train -= x_train_mean
#     x_test -= x_train_mean


if __name__ == '__main__':
    main()
