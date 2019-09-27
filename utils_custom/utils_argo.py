import numpy as np
import tensorflow as tf

from hparms import *

IMAGE_TO_REAL_SCALE = 0.2


def agent_cord_to_image_cord(pos: np.ndarray):
    pos = pos / IMAGE_TO_REAL_SCALE
    if len(pos.shape) == 1:
        pos[1] = pos[1] * -1
    else:
        pos[:, 1] = pos[:, 1] * -1
    pos = pos + [IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2] + 0.5

    return pos.astype(np.int)


def FDE_3S(y_true, y_pred):
    # y_pred_final = y_pred[:, 58:60]
    # y_true_final = y_true[:, 58:60]
    y_pred_final = tf.gather(y_pred, [29, 59], axis=1)
    y_true_final = tf.gather(y_true, [29, 59], axis=1)
    df = y_true_final - y_pred_final

    return tf.math.reduce_mean(tf.math.sqrt(tf.reduce_sum(df * df, axis=1)))


def FDE_2S(y_true, y_pred):
    # y_pred_final = y_pred[:, 58:60]
    # y_true_final = y_true[:, 58:60]
    y_pred_final = tf.gather(y_pred, [19, 49], axis=1)
    y_true_final = tf.gather(y_true, [19, 49], axis=1)
    df = y_true_final - y_pred_final

    return tf.math.reduce_mean(tf.math.sqrt(tf.reduce_sum(df * df, axis=1)))


def FDE_1S(y_true, y_pred):
    # y_pred_1s = y_pred[:, 18:20]
    # y_true_1s = y_true[:, 18:20]
    y_pred_1s = tf.gather(y_pred, [9, 39], axis=1)
    y_true_1s = tf.gather(y_true, [9, 39], axis=1)
    df = y_true_1s - y_pred_1s

    return tf.math.reduce_mean(tf.math.sqrt(tf.reduce_sum(df * df, axis=1)))


def ADE_3S(y_true, y_pred):
    # ind = [2*i for i in range(30)]
    # true_x = tf.gather(y_true, ind, axis=1)
    # pred_x = tf.gather(y_pred, ind, axis=1)

    # ind = [2*i+1 for i in range(30)]
    # true_y = tf.gather(y_true, ind, axis=1)
    # pred_y = tf.gather(y_pred, ind, axis=1)

    true_x = y_true[:, 0:30]
    pred_x = y_pred[:, 0:30]
    true_y = y_true[:, 30:60]
    pred_y = y_pred[:, 30:60]

    return tf.math.reduce_mean(
        tf.math.reduce_mean(tf.math.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2), axis=1))


def ADE_1S(y_true, y_pred):
    # ind = [2*i for i in range(10)]
    # true_x = tf.gather(y_true, ind, axis=1)
    # pred_x = tf.gather(y_pred, ind, axis=1)
    #
    # ind = [2*i+1 for i in range(10)]
    # true_y = tf.gather(y_true, ind, axis=1)
    # pred_y = tf.gather(y_pred, ind, axis=1)

    true_x = y_true[:, 0:10]
    pred_x = y_pred[:, 0:10]
    true_y = y_true[:, 30:40]
    pred_y = y_pred[:, 30:40]

    return tf.math.reduce_mean(
        tf.math.reduce_mean(tf.math.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2), axis=1))


def ADE_FDE_loss(y_true, y_pred):

    true_x = y_true[:, 0:30]
    pred_x = y_pred[:, 0:30]
    true_y = y_true[:, 30:60]
    pred_y = y_pred[:, 30:60]

    de = tf.math.reduce_mean(tf.math.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2), axis=0)

    fde_1s = de[9]
    fde_2s = de[19]
    fde_3s = de[29]
    ade_3s = tf.math.reduce_mean(de)

    return ade_3s + 0.3 * fde_1s + 0.63 * fde_2s + 0.96 * fde_3s


def ADE_3S_array(y_true: np.ndarray, y_pred: np.ndarray):

    true_x = y_true[:, 0:30]
    pred_x = y_pred[:, 0:30]
    true_y = y_true[:, 30:60]
    pred_y = y_pred[:, 30:60]

    return np.mean(np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2), axis=1)