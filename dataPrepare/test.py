import os
import tensorflow as tf

DATA_DIR = "../../data/"

training_files = tf.gfile.Glob(os.path.join(DATA_DIR, 'train', '*', '*.png'))


