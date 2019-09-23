import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from hparms import *
from math import pi
from argoPrepare.load_tfrecord_argo import input_fn
from netTrain.GMM.net_model import ResNet50V2_gmm

M = NUM_GAUSSIAN_COMPONENT
T = FUTURE_TIME_STEP


def log_likelihood_loss(y_true, y_pred):
    mu_x, mu_y, log_var_x, log_var_y, rho, w = tf.split(y_pred, [M * T, M * T, M * T, M * T, M * T, M * T], axis=1)

    ind = [2 * i for i in range(T)]
    true_x = tf.gather(y_true, ind, axis=1)
    ind = [2 * i + 1 for i in range(T)]
    true_y = tf.gather(y_true, ind, axis=1)

    mu_x = tf.reshape(mu_x, [-1, M, T])
    mu_y = tf.reshape(mu_y, [-1, M, T])

    log_var_x = tf.clip_by_value(log_var_x, clip_value_min=-5.0, clip_value_max=5)
    log_var_y = tf.clip_by_value(log_var_y, clip_value_min=-5.0, clip_value_max=5)
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

    :param y_pred: [1*(6*M*T)]
    :return:
    """
    mu_x, mu_y, log_var_x, log_var_y, rho, w = np.split(y_pred, 6, axis=1)

    mu_x = np.reshape(mu_x, [M, T])
    mu_y = np.reshape(mu_y, [M, T])

    var_x = np.reshape(np.exp(log_var_x), [M, T])
    var_y = np.reshape(np.exp(log_var_y), [M, T])

    rho = rho / (np.abs(rho) + 1)
    rho = np.reshape(rho, [M, T])

    w = np.reshape(w, [M, T])
    w = np.exp(w) / (np.sum(np.exp(w), axis=0).reshape(1, T))
    # w[0, 0] = 0.5
    # w[1, 0] = 0.5

    xx = []
    yy = []
    cc = []
    for t in range(1):
        for m in range(M):
            var = max(var_x[m, t], var_y[m, t])
            if var > 5:
                continue
            for x in np.arange(mu_x[m, t]-2*var, mu_x[m, t]+2*var, 0.1):
                for y in np.arange(mu_y[m, t]-2*var, mu_y[m, t]+2*var, 0.1):
                    true_x = x * np.ones([M, 1])
                    true_y = y * np.ones([M, 1])

                    diff_x = true_x - mu_x[:, t].reshape(-1, 1)
                    diff_y = true_y - mu_y[:, t].reshape(-1, 1)

                    exponent = -1 / 2 / (1 - rho[:, t].reshape(-1, 1) ** 2) * \
                               (diff_x ** 2 / var_x[:, t].reshape(-1, 1) ** 2
                                + diff_y ** 2 / var_y[:, t].reshape(-1, 1) ** 2
                                - 2*rho[:,t].reshape(-1,1)*diff_x*diff_y/var_x[:,t].reshape(-1,1)/var_y[:,t].reshape(-1,1))

                    p_i = w[:, t].reshape(-1, 1) \
                          / (2 * pi * var_x[:, t].reshape(-1, 1) * var_y[:, t].reshape(-1, 1) * np.sqrt(1 - rho[:, t].reshape(-1, 1) ** 2)) \
                          * np.exp(exponent)
                    p = np.sum(p_i)

                    if p < 4e-2:
                        continue
                    xx.append(x)
                    yy.append(y)
                    cc.append(p)

    print(sum(cc))
    plt.scatter(xx, yy, c=cc, s=50)
    plt.show()


def main():
    # y_pred = np.random.rand(1, 6*M*T)
    # gmm_distribution(y_pred)
    sess = tf.Session()
    data_dir = ['../../../data/argo/forecasting/val/tf_record/']
    dataset = input_fn(is_training=False, data_dir=data_dir, batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, traj_batch = iterator.get_next()
    model = ResNet50V2_gmm(weights='../../../logs/GMM/checkpoints/20190919-124342weights010.h5',
                           input_img_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                           input_ptraj_shape=(PAST_TIME_STEP*2, ),
                           node_num=2048,
                           gmm_comp=NUM_GAUSSIAN_COMPONENT,
                           time_steps=FUTURE_TIME_STEP)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=log_likelihood_loss)

    im, la = sess.run([image_batch, traj_batch])
    y = model.predict(im)
    gmm_distribution(y)
    pass


if __name__ == '__main__':
    main()
