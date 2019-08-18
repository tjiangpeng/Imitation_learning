import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils.load_tfrecord import input_fn
from matplotlib import pyplot as plt

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
NUM_CHANNELS = 3

keras.backend.clear_session()
sess = tf.Session()

data_dir = ['../../data/2019_08_07/', '../../data/2019_08_10/', '../../data/2019_08_12/', '../../data/2019_08_12_2/']
# data_dir = ['../../data/2019_08_12_2/']
dataset = input_fn(is_training=True, data_dir=data_dir, batch_size=1, num_epochs=1)
iterator = dataset.make_one_shot_iterator()
image_batch, traj_batch = iterator.get_next()

num = 0
image_mean = np.zeros(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
while True:
    try:
        im, la = sess.run([image_batch, traj_batch])
        image_mean = image_mean + im[0]
        num += 1
        print(num)
    except tf.errors.OutOfRangeError:
        break

image_mean = image_mean / num
plt.imshow(image_mean)
plt.show()
np.save('image_mean.npy', image_mean)
sess.close()
