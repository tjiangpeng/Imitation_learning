import os

import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from argoPrepare.load_tfrecord_argo import input_fn
from netTrain.ResNet.net_model import ResNet50V2, ResNet50V2_fc
from hparms import *
from utils_custom.metrics import FDE_1S, FDE_3S, ADE_1S, ADE_3S

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # specify which GPU(s) to be used


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    dataset = input_fn(is_training=False, data_dir=['../../../data/argo/forecasting/val/tf_record/'], batch_size=16)
    iterator = dataset.make_one_shot_iterator()
    image_batch, traj_batch = iterator.get_next()

    # Model
#    model = ResNet50V2(include_top=True, weights='../../../logs/ResNet/checkpoints/20190911-083837weights018.h5',
#                       input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
#                       classes=NUM_TIME_SEQUENCE*2)
    model = ResNet50V2_fc(weights='../../../logs/ResNet/checkpoints/20190912-095604weights055.h5',
                          input_img_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                          input_ptraj_shape=(PAST_TIME_STEP*2, ),
                          node_num=2048,
                          classes=NUM_TIME_SEQUENCE*2)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='mse',
                  metrics=[ADE_1S, FDE_1S, ADE_3S, FDE_3S])

    # Evaluate
    scores = model.evaluate(dataset, verbose=1, steps=2500)
    print("%s: %.2f" % (model.metrics_names[1], scores[1]))
    print("%s: %.2f" % (model.metrics_names[2], scores[2]))
    print("%s: %.2f" % (model.metrics_names[3], scores[3]))
    print("%s: %.2f" % (model.metrics_names[4], scores[4]))
#    print("%s: %.2f" % (model.metrics_names[5], scores[5]))

    # outVideo = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (101, 115))
    for i in range(10000):
        im, la = sess.run([image_batch, traj_batch])
        y = model.predict(im)
        # print(y[0])
        image = im["input_1"][0]
        image = image * 255.0
        image = image.astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        la[0] = la[0] / 0.2
        y[0] = y[0] / 0.2
        for ind in range(NUM_TIME_SEQUENCE):
            pos_gt = la[0][2*ind:2*ind+2]
            pos_pred = y[0][2*ind:2*ind+2]

            pos_gt[1] = pos_gt[1] * -1
            pos_gt = pos_gt + [IMAGE_WIDTH/2, IMAGE_HEIGHT/2]
            pos_gt = pos_gt.astype(np.int)

            pos_pred[1] = pos_pred[1] * -1
            pos_pred = pos_pred + [IMAGE_WIDTH/2, IMAGE_HEIGHT/2]
            pos_pred = pos_pred.astype(np.int)

            if 0 <= pos_gt[0] < IMAGE_HEIGHT and 0 <= pos_gt[1] < IMAGE_HEIGHT:
                image[pos_gt[1]][pos_gt[0]] = (0, 0, 255)
            if 0 <= pos_pred[0] < IMAGE_HEIGHT and 0 <= pos_pred[1] < IMAGE_HEIGHT:
                image[pos_pred[1]][pos_pred[0]] = (255, 255, 255)
        cv2.imshow('image', image)

        # outVideo.write(cropImg)

        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break
        print(i)
    # outVideo.release()
    sess.close()

    pass


if __name__ == '__main__':
    main()