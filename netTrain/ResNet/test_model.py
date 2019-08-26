import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils.load_tfrecord import input_fn
from netTrain.ResNet.net_model import ResNet50V2
from hparms import *


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    dataset = input_fn(is_training=False, data_dir=['../../../data/2019_08_14/'], batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, traj_batch = iterator.get_next()

    # Model
    model = ResNet50V2(include_top=True, weights='../../../logs/ResNet/checkpoints/20190818-191529weights056.h5',
                       input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                       classes=NUM_TIME_SEQUENCE*2)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['mae'])

    # Evaluate
    # scores = model.evaluate(dataset, verbose=1, steps=10000)
    # print("%s: %.2f" % (model.metrics_names[1], scores[1]))

    outVideo = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (101, 115))
    for i in range(10000):
        im, la = sess.run([image_batch, traj_batch])
        y = model.predict(im)
        # print(y[0])
        image = im[0]
        image = image * 255.0
        image = image.astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        la[0] = la[0] + 0.5
        y[0] = y[0] + 0.5
        for ind in range(NUM_TIME_SEQUENCE):
            pos_gt = la[0][2*ind:2*ind+2].astype(np.int)
            pos_pred = y[0][2*ind:2*ind+2].astype(np.int)

            image[pos_gt[1]][pos_gt[0]] = (255, 255, 255)
            image[pos_pred[1]][pos_pred[0]] = (0, 0, 0)
        cv2.imshow('image', image)

        cropImg = image[203:318, 142:243]
        cv2.imshow('cropped image', cropImg)
        outVideo.write(cropImg)

        cv2.waitKey(1)
        # while True:
        #     k = cv2.waitKey(1)
        #     if k == 27:
        #         break
        print(i)
    outVideo.release()
    sess.close()

    pass


if __name__ == '__main__':
    main()