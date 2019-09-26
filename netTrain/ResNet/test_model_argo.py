import os

import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
# from argoPrepare.load_tfrecord_argo import input_fn
from argoData.load_tfrecord_argo import input_fn
from netTrain.ResNet.net_model import ResNet50V2, ResNet50V2_fc
from hparms import *
from utils_custom.utils_argo import FDE_1S, FDE_3S, ADE_1S, ADE_3S, agent_cord_to_image_cord

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # specify which GPU(s) to be used


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    dataset = input_fn(is_training=True, data_dir=['../../../data/argo/forecasting/val/tf_record_4_channel/'], batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, traj_batch = iterator.get_next()

    # Model
    model = ResNet50V2(include_top=True, weights=None,
                       input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                       classes=FUTURE_TIME_STEP*2)
    # model = ResNet50V2_fc(weights=None,
    #                       input_img_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
    #                       input_ptraj_shape=(PAST_TIME_STEP*2, ),
    #                       node_num=2048,
    #                       classes=NUM_TIME_SEQUENCE*2)

    # model = keras.utils.multi_gpu_model(model, gpus=4)
    model.load_weights('../../../logs/ResNet/checkpoints/20190925-201705weights028.h5')
    # model.load_weights('new_model.h5')

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='mse',
                  metrics=[ADE_1S, FDE_1S, ADE_3S, FDE_3S])

    # Evaluate
    # scores = model.evaluate(dataset, verbose=1, steps=2500)
    # print("%s: %.2f" % (model.metrics_names[1], scores[1]))
    # print("%s: %.2f" % (model.metrics_names[2], scores[2]))
    # print("%s: %.2f" % (model.metrics_names[3], scores[3]))
    # print("%s: %.2f" % (model.metrics_names[4], scores[4]))
    # print("%s: %.2f" % (model.metrics_names[5], scores[5]))

    for i in range(10000):
        im, la = sess.run([image_batch, traj_batch])
        y = model.predict(im)
        # print(y[0])
        image = im["input_1"][0]
        image = image * 255.0
        image = image.astype(np.uint8)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for ind in range(FUTURE_TIME_STEP):
            pos_gt = la[0, [ind, FUTURE_TIME_STEP+ind]]
            pos_pred = y[0, [ind, FUTURE_TIME_STEP+ind]]

            pos_gt = agent_cord_to_image_cord(pos_gt)
            pos_pred = agent_cord_to_image_cord(pos_pred)

            if 0 <= pos_gt[0] < IMAGE_HEIGHT and 0 <= pos_gt[1] < IMAGE_HEIGHT:
                image[pos_gt[1], pos_gt[0], 1] = 100
            if 0 <= pos_pred[0] < IMAGE_HEIGHT and 0 <= pos_pred[1] < IMAGE_HEIGHT:
                image[pos_pred[1], pos_pred[0], 1] = 255
        cv2.imshow('image', image[:, :, [0, 1, 3]])

        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break
        print(i)
    sess.close()

    pass


if __name__ == '__main__':
    main()