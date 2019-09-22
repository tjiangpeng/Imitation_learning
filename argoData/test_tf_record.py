import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from argoData.load_tfrecord_argo import input_fn
from hparms import *


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    data_dir = ['../../data/argo/forecasting/sample/tf_record/']

    dataset = input_fn(is_training=False, data_dir=data_dir, batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, future_traj_batch = iterator.get_next()

    while True:
        im, ftraj = sess.run([image_batch, future_traj_batch])
        im = im["input_1"][0] * 255.0
        im = im.astype(np.uint8)

        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        cv2.imshow('image', image)
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break


if __name__ == '__main__':
    main()
