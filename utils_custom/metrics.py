import tensorflow as tf


def FDE_3S(y_true, y_pred):
    # y_pred_final = y_pred[:, 58:60]
    # y_true_final = y_true[:, 58:60]
    y_pred_final = tf.gather(y_pred, [29, 59], axis=1)
    y_true_final = tf.gather(y_true, [29, 59], axis=1)
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