import os
import math
from random import shuffle
# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import cv2
import argparse
import tensorflow as tf
import numpy as np

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================


# ==============================================================================
# -- Functuion -----------------------------------------------------------------
# ==============================================================================

def _check_or_create_dir(directory):
    """Check if drectory exists otherwise create it"""
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)

def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
      value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(past_traj, future_traj, center_lines, surr_past_pos):
    """Build an Example proto for an example"""

    img_bytes = open('temp.png', 'rb').read()

    clines_num = len(center_lines)
    past_traj = np.reshape(np.transpose(past_traj), [-1, ]).tolist()
    future_traj = np.reshape(np.transpose(future_traj), [-1, ]).tolist()

    cline_points_list = [i.shape[0] for i in center_lines]
    clines_array = 10000.0 * np.ones([clines_num, max(cline_points_list), 2])
    for ind, line in enumerate(center_lines):
        clines_array[ind][0:line.shape[0], :] = line

    example = tf.train.Example(features=tf.train.Features(feature={
        'map': _bytes_feature(img_bytes),
        'past_traj': _float_feature(past_traj),
        'future_traj': _float_feature(future_traj),
        'center_lines/data': _bytes_feature(clines_array.tostring()),
        'center_lines/num': _int64_feature(clines_num),
        'surr_past_pos': _bytes_feature(surr_past_pos.tostring())
    }))

    return example

