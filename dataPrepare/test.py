import os
import tensorflow as tf

from absl import app
from absl import flags

flags.DEFINE_string(
    'project', None, 'Google cloud project id for uploading the dataset.')
flags.DEFINE_string(
    'gcs_output_path', None, 'GCS path for uploading the dataset.')
flags.DEFINE_string(
    'local_scratch_dir', None, 'Scratch directory path for temporary files.')
flags.DEFINE_string(
    'raw_data_dir', None, 'Directory path for raw Imagenet dataset. '
    'Should have train and validation subdirectories inside it.')
flags.DEFINE_string(
    'imagenet_username', None, 'Username for Imagenet.org account')
flags.DEFINE_string(
    'imagenet_access_key', None, 'Access Key for Imagenet.org account')
flags.DEFINE_boolean(
    'gcs_upload', True, 'Set to false to not upload to gcs.')

FLAGS = flags.FLAGS

a = 1

