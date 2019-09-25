import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
import argoData
from argoData.load_tfrecord_argo import input_fn, cur_num
from hparms import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main():
    keras.backend.clear_session()
    sess = tf.Session()

    data_dir = ['../../data/argo/forecasting/sample/tf_record_4_channel/']

    dataset = input_fn(is_training=False, data_dir=data_dir, batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, future_traj_batch = iterator.get_next()

    while True:
        im, ftraj = sess.run([image_batch, future_traj_batch])
        # print(im["input_2"][0])
        # print("++++++++++++++")
        # print(ftraj[0])

        im = im["input_1"][0] * 255.0
        im = im.astype(np.uint8)
        # image = cv2.cvtColor(im[:, :, 0:3], cv2.COLOR_BGR2RGB)
        image = im[:, :, 0]

        past_traj = im[:, :, 1]
        clines = im[:, :, 2]
        surr = im[:, :, 3]
        cv2.imshow('image', image)
        cv2.imshow('past_traj', past_traj)
        cv2.imshow('clines', clines)
        cv2.imshow('surr', surr)
        # while True:
        #     k = cv2.waitKey(1)
        #     if k == 27:
        #         break

        print(argoData.load_tfrecord_argo.cur_num)


if __name__ == '__main__':
    main()
