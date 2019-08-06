# from tensorflow import keras
# from tensorflow.keras import layers
# import tensorflow as tf
# import numpy as np
# from keras.datasets import cifar10

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# cifar10.load_data()