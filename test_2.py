import tensorflow as tf
import numpy as np

sess = tf.Session()
t = np.random.rand(4, 60)
p = np.random.rand(4, 60)
y_true = tf.constant(t)
y_pred = tf.constant(p)

ind = [2 * i for i in range(30)]
true_x = tf.gather(y_true, ind, axis=1)
pred_x = tf.gather(y_pred, ind, axis=1)

ind = [2 * i + 1 for i in range(30)]
true_y = tf.gather(y_true, ind, axis=1)
pred_y = tf.gather(y_pred, ind, axis=1)

re = tf.math.reduce_mean(tf.math.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2), axis=1)

print(sess.run(re))

ind = [2 * i for i in range(30)]
x_true = t[:, ind]
x_pred = p[:, ind]
ind = [2 * i + 1 for i in range(30)]
y_true = t[:, ind]
y_pred = p[:, ind]

e = np.mean(np.mean(np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2), axis=1))
print(e)

