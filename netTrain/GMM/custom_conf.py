import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from hparms import *
from math import pi

M = NUM_GAUSSIAN_COMPONENT
T = NUM_TIME_SEQUENCE


def log_likelihood_loss(y_true, y_pred):
    mu_x, mu_y, log_var_x, log_var_y, rho, w = tf.split(y_pred, [M * T, M * T, M * T, M * T, M * T, M * T], axis=1)

    ind = [2 * i for i in range(T)]
    true_x = tf.gather(y_true, ind, axis=1)
    ind = [2 * i + 1 for i in range(T)]
    true_y = tf.gather(y_true, ind, axis=1)

    mu_x = tf.reshape(mu_x, [-1, M, T])
    mu_y = tf.reshape(mu_y, [-1, M, T])

    log_var_x = tf.clip_by_value(log_var_x, clip_value_min=-3.0, clip_value_max=4)
    log_var_y = tf.clip_by_value(log_var_y, clip_value_min=-3.0, clip_value_max=4)
    var_x = tf.reshape(tf.exp(log_var_x), [-1, M, T])
    var_y = tf.reshape(tf.exp(log_var_y), [-1, M, T])

    rho = keras.activations.softsign(rho)
    rho = tf.reshape(rho, [-1, M, T])

    w = tf.reshape(w, [-1, M, T])
    w = keras.activations.softmax(w, axis=1)

    diff_x = tf.reshape(tf.tile(true_x, tf.constant([1, M])), [-1, M, T]) - mu_x
    diff_y = tf.reshape(tf.tile(true_y, tf.constant([1, M])), [-1, M, T]) - mu_y

    exponent = -1 / 2 / (1 - rho ** 2) * (
                diff_x ** 2 / var_x ** 2 + diff_y ** 2 / var_y ** 2 - 2 * rho * diff_x * diff_y / var_x / var_y)
    exponent = tf.clip_by_value(exponent, clip_value_min=-50, clip_value_max=0)

    f_i = w * 1 / (var_x * var_y * tf.math.sqrt(1 - rho ** 2)) * tf.exp(exponent)
    f_sum = tf.reduce_sum(f_i, axis=1)

    likelihood = keras.activations.relu(-tf.log(f_sum), max_value=1000)
    loss_mat = tf.reduce_mean(likelihood, axis=1)

    return tf.reduce_mean(loss_mat, axis=0)


def gmm_distribution(y_pred):
    """

    :param y_pred: [n*(6*M*T)]
    :return:
    """
    n = y_pred.shape[0]
    mu_x, mu_y, log_var_x, log_var_y, rho, w = np.split(y_pred, 6, axis=1)

    mu_x = np.reshape(mu_x, [-1, M, T])
    mu_y = np.reshape(mu_y, [-1, M, T])

    var_x = np.reshape(np.exp(log_var_x), [-1, M, T])
    var_y = np.reshape(np.exp(log_var_y), [-1, M, T])

    rho = rho / (rho + 1)
    rho = np.reshape(rho, [-1, M, T])
    rho = np.zeros([n, M, T])

    w = np.reshape(w, [-1, M, T])
    w = np.exp(w) / (np.sum(np.exp(w), axis=1).reshape(-1, 1, T))

    xx = []
    yy = []
    cc = []
    for x in np.arange(-10, 10, 0.1):
        for y in np.arange(-10, 10, 0.1):
            true_x = x * np.ones([n, M, T])
            true_y = y * np.ones([n, M, T])

            diff_x = true_x - mu_x
            diff_y = true_y - mu_y

            exponent = -1/2/(1-rho**2) * \
                       (diff_x**2/var_x**2 + diff_y**2/var_y**2 - 2*rho*diff_x*diff_y/var_x/var_y)

            p_i = w / (2*pi*var_x*var_y*np.sqrt(1-rho**2)) * np.exp(exponent)
            p = np.sum(p_i, axis=1)

            xx.append(x)
            yy.append(y)
            cc.append(p[0][0])
    print(sum(cc))
    plt.scatter(xx, yy, c=cc)
    plt.show()

def main():
    y_pred = np.random.rand(5, 6*M*T)
    gmm_distribution(y_pred)

if __name__ == '__main__':
    main()
