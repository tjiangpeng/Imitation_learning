import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils.load_tfrecord import input_fn
from netTrain.VggNet.net_model import VGG16_FL
from hparms import *

NUM_HIDDEN_NODES = 2500


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    dataset = input_fn(is_training=False, data_dir=['../../../data/2019_08_14/'], batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, traj_batch = iterator.get_next()

    # Model
    model = VGG16_FL(num_node=NUM_HIDDEN_NODES, num_time_sequence=FUTURE_TIME_STEP,
                     weight='logs/checkpoints/20190815-171720weights0062.h5')

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['mae'])

    # Evaluate
    # scores = model.evaluate(dataset, verbose=1, steps=1250)
    # print("%s: %.2f" % (model.metrics_names[1], scores[1]))

    # image_mean = np.load("../../utils/image_mean.npy")
    # outVideo = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (101, 115))
    for i in range(6000):
        im, la = sess.run([image_batch, traj_batch])
        y = model.predict(im)

        image = im[0]
        image = image * 255.0
        image = image.astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        la[0] = la[0] + 0.5
        y[0] = y[0] + 0.5
        for ind in range(FUTURE_TIME_STEP):
            pos_gt = la[0][2*ind:2*ind+2].astype(np.int)
            pos_pred = y[0][2*ind:2*ind+2].astype(np.int)

            image[pos_gt[1]][pos_gt[0]] = (255, 255, 255)
            image[pos_pred[1]][pos_pred[0]] = (0, 0, 0)
        cv2.imshow('image', image)

        cropImg = image[203:318, 142:243]
        cv2.imshow('cropped image', cropImg)
        # outVideo.write(cropImg)

        cv2.waitKey(1)
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