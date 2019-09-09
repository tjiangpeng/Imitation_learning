import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from argoPrepare.load_tfrecord_argo import input_fn
from hparms import *


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    # data_dir = ['../../data/argo/argoverse-tracking/train1_tf_record/',
    #             '../../data/argo/argoverse-tracking/train2_tf_record/',
    #             '../../data/argo/argoverse-tracking/train3_tf_record/',
    #             '../../data/argo/argoverse-tracking/train4_tf_record/']

    data_dir = ['../../data/argo/forecasting/train/tf_record/']

    dataset = input_fn(is_training=True, data_dir=data_dir, batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, past_traj_batch, future_traj_batch = iterator.get_next()

    i = 1
    while True:
        im, ptraj, ftraj = sess.run([image_batch, past_traj_batch, future_traj_batch])
        im = im * 255.0
        im = im.astype(np.uint8)

        image = cv2.cvtColor(im[0], cv2.COLOR_BGR2RGB)

        # print(la)
        ptraj = ptraj / 0.2
        ftraj = ftraj / 0.2
        for ind in range(PAST_TIME_STEP):
            pos_gt = ptraj[0][2 * ind:2 * ind + 2]
            pos_gt[1] = pos_gt[1] * -1
            pos_gt = pos_gt + [IMAGE_WIDTH/2, IMAGE_HEIGHT/2]
            pos_gt = pos_gt.astype(np.int)
            image[pos_gt[1]][pos_gt[0]] = (255, 255, 255)

        for ind in range(NUM_TIME_SEQUENCE):
            pos_gt = ftraj[0][2 * ind:2 * ind + 2]
            pos_gt[1] = pos_gt[1] * -1
            pos_gt = pos_gt + [IMAGE_WIDTH/2, IMAGE_HEIGHT/2]
            pos_gt = pos_gt.astype(np.int)
            image[pos_gt[1]][pos_gt[0]] = (255, 255, 255)

        cv2.imshow('image', image)
        cv2.waitKey(1)
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break
        print(i)
        i += 1
    sess.close()


if __name__ == '__main__':
    main()
