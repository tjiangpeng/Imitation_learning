import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from argoPrepare.load_tfrecord_argo import input_fn
from netTrain.ResNet.net_model import ResNet50V2
from hparms import *


def FDE_3S(y_true, y_pred):
    y_pred_final = y_pred[:, 58:60]
    y_true_final = y_true[:, 58:60]
    df = y_true_final - y_pred_final

    return tf.math.reduce_mean(tf.math.sqrt(tf.reduce_sum(df * df, axis=1)))


def FDE_1S(y_true, y_pred):
    y_pred_1s = y_pred[:, 18:20]
    y_true_1s = y_true[:, 18:20]
    df = y_true_1s - y_pred_1s

    return tf.math.reduce_mean(tf.math.sqrt(tf.reduce_sum(df * df, axis=1)))


def ADE_3S(y_true, y_pred):
    ind = [2*i for i in range(30)]
    true_x = tf.gather(y_true, ind, axis=1)
    pred_x = tf.gather(y_pred, ind, axis=1)

    ind = [2*i+1 for i in range(30)]
    true_y = tf.gather(y_true, ind, axis=1)
    pred_y = tf.gather(y_pred, ind, axis=1)

    return tf.math.reduce_mean(
        tf.math.reduce_mean(tf.math.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2), axis=1))


def ADE_1S(y_true, y_pred):
    ind = [2*i for i in range(10)]
    true_x = tf.gather(y_true, ind, axis=1)
    pred_x = tf.gather(y_pred, ind, axis=1)

    ind = [2*i+1 for i in range(10)]
    true_y = tf.gather(y_true, ind, axis=1)
    pred_y = tf.gather(y_pred, ind, axis=1)

    return tf.math.reduce_mean(
        tf.math.reduce_mean(tf.math.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2), axis=1))


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    dataset = input_fn(is_training=False, data_dir=['../data/argo/forecasting/sample/tf_record/'], batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, traj_batch = iterator.get_next()

    # Model
    model = ResNet50V2(include_top=True, weights='../logs/ResNet/checkpoints/20190910-112359weights042.h5',
                       input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                       classes=NUM_TIME_SEQUENCE*2)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['mae', FDE_1S, FDE_3S, ADE_3S, ADE_1S])

    # Evaluate
    scores = model.evaluate(dataset, verbose=1, steps=4)
    print("%s: %.2f" % (model.metrics_names[2], scores[2]))
    print("%s: %.2f" % (model.metrics_names[3], scores[3]))
    print("%s: %.2f" % (model.metrics_names[4], scores[4]))
    print("%s: %.2f" % (model.metrics_names[5], scores[5]))

    # outVideo = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (101, 115))
    fde_error = []
    for i in range(10000):
        im, la = sess.run([image_batch, traj_batch])
        y = model.predict(im)
        # print(y[0])
        image = im[0]
        image = image * 255.0
        image = image.astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ####################################
        df = la[0][-2:] - y[0][-2:]
        # fde_error.append(np.sqrt(np.sum(df*df)))

        ind = [2 * i for i in range(10)]
        x_true = la[0][ind]
        x_pred = y[0][ind]
        ind = [2*i+1 for i in range(10)]
        y_true = la[0][ind]
        y_pred = y[0][ind]

        e = np.mean(np.sqrt((x_true-x_pred)**2 + (y_true-y_pred)**2))
        fde_error.append(e)
        print(fde_error)
        print(sum(fde_error)/len(fde_error))



        ####################################
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