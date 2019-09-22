import os
import glob
import tensorflow as tf
import numpy as np
from random import shuffle

from hparms import *
from utils_custom.utils_argo import agent_cord_to_image_cord
###############################################################################
# Constant
###############################################################################



###############################################################################
# Data processing
###############################################################################


def render_past_traj(map, past_traj):

    for ind in range(PAST_TIME_STEP):
        pos = np.array([past_traj[ind], past_traj[ind+PAST_TIME_STEP]])
        pos = agent_cord_to_image_cord(pos)

        if 0 <= pos[0] < IMAGE_HEIGHT and 0 <= pos[1] < IMAGE_HEIGHT:
            map[pos[1]][pos[0]] = (0.0, 0.0, 255.0)
    return map


def render_future_traj(img, future_traj):

    for ind in range(FUTURE_TIME_STEP):
        pos = np.array([future_traj[ind], future_traj[ind+FUTURE_TIME_STEP]])
        pos = agent_cord_to_image_cord(pos)

        if 0 <= pos[0] < IMAGE_HEIGHT and 0 <= pos[1] < IMAGE_HEIGHT:
            img[pos[1]][pos[0]] = (255.0, 0.0, 0.0)
    return img


def render_center_lines(img, center_lines, clines_num):
    clines_num = int(clines_num[0])
    center_lines = np.reshape(center_lines, [clines_num, -1, 2])
    for i in range(clines_num):
        cline = center_lines[i]
        for j in range(cline.shape[0]):
            if cline[j, 0] != -1000:
                pos = agent_cord_to_image_cord(cline[j, :])
                if 0 <= pos[0] < IMAGE_HEIGHT and 0 <= pos[1] < IMAGE_HEIGHT:
                    img[pos[1]][pos[0]] = (255.0, 255.0, 255.0)
    return img


def render_surr(img, surr_past_pos):
    for i in range(surr_past_pos.shape[0]):
        for ind in range(SURR_TIME_STEP+1):
            if surr_past_pos[i, ind] != -1000:
                pos = np.array([surr_past_pos[i, ind], surr_past_pos[i, ind+SURR_TIME_STEP+1]])
                pos = agent_cord_to_image_cord(pos)
                if 0 <= pos[0] < IMAGE_HEIGHT and 0 <= pos[1] < IMAGE_HEIGHT:
                    img[pos[1]][pos[0]] = (255.0, 255.0, 0.0)
    return img


def get_filenames(is_training, data_dir):
    """Return filenames for dataset. (for argoverse dataset)"""
    filenames = []
    for dir in data_dir:
        filenames = filenames + glob.glob(os.path.join(dir, '*_tf_record'))
    if is_training:
        shuffle(filenames)
    return filenames


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
    """
    # Dense features in Example proto.
    feature_map = {
        'map': tf.FixedLenFeature([], dtype=tf.string,
                                  default_value=''),
        'past_traj': tf.FixedLenFeature([PAST_TIME_STEP * 2], dtype=tf.float32),
        'future_traj': tf.FixedLenFeature([FUTURE_TIME_STEP * 2], dtype=tf.float32),
        'center_lines/data': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
        'center_lines/num': tf.FixedLenFeature([1], dtype=tf.int64),
        'surr_past_pos': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value='')
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    return features


def parse_record(raw_record):
    """Parses a record containing a training example of an image.

    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps.

    Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

    Returns:
    Tuple with processed image tensor and future trajectory tensor.
    """
    features = _parse_example_proto(raw_record)

    past_traj = features['past_traj']
    future_traj = features['future_traj']
    clines_num = features['center_lines/num']

    center_lines = tf.decode_raw(features['center_lines/data'], tf.float64)
    center_lines = tf.cast(center_lines, tf.float32)

    surr_past_pos = tf.decode_raw(features['surr_past_pos'], tf.float64)
    surr_past_pos = tf.cast(surr_past_pos, tf.float32)
    surr_past_pos = tf.reshape(surr_past_pos, shape=[-1, (SURR_TIME_STEP+1)*2])

    map = tf.image.decode_png(features['map'], channels=NUM_CHANNELS)
    map.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    map = tf.cast(map, tf.float32)

    image = tf.py_func(render_past_traj, [map, past_traj], tf.float32)
    image = tf.py_func(render_future_traj, [image, future_traj], tf.float32)
    image = tf.py_func(render_center_lines, [image, center_lines, clines_num], tf.float32)
    image = tf.py_func(render_surr, [image, surr_past_pos], tf.float32)
    # normalize image
    image = image / 255.0

    return {"input_1": image, "input_2": past_traj}, future_traj


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input function which provides batches for train or eval.

    Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

    Returns:
    A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        dataset = dataset.apply(tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=10))
        dataset = dataset.prefetch(buffer_size=batch_size)

        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=2048)
    else:
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.prefetch(buffer_size=batch_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parse_record)
    dataset = dataset.batch(batch_size)

    return dataset

