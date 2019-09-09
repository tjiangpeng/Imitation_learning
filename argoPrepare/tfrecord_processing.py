import os
import math
from random import shuffle
# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import cv2
import argparse
import tensorflow as tf

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


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(past_traj, future_traj):
    """Build an Example proto for an example"""
    # img = cv2.imread('temp.png')
    # cv2.imshow('img', img)
    # cv2.waitKey(1)

    img_bytes = open('temp.png', 'rb').read()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(img_bytes),
        'past_traj': _bytes_feature(past_traj.tostring()),
        'future_traj': _bytes_feature(future_traj.tostring())
    }))

    return example

