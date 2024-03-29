import os
import glob
import tensorflow as tf
import numpy as np
from random import shuffle
from hparms import *
###############################################################################
# Constant
###############################################################################

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 4
_NUM_VALIDATION_FILES = 1
_SHUFFLE_BUFFER = 10000
# IMAGE_MEAN_ARRAY = np.load("/home/shaojun/jiangpeng/CARLA/CARLA_0_9_6/PythonAPI/Imitation_learning/utils_custom/image_mean.npy")
###############################################################################
# Data processing
###############################################################################


def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        shards_taken = []
        shards_names = [[], [], [], []]
        for dir in data_dir:
            shards_names[0] = shards_names[0] + glob.glob(os.path.join(dir, 'train', '*bend-road*'))
            shards_names[1] = shards_names[1] + glob.glob(os.path.join(dir, 'train', '*red-light*'))
            shards_names[2] = shards_names[2] + glob.glob(os.path.join(dir, 'train', '*front-vehicle*'))
            shards_names[3] = shards_names[3] + glob.glob(os.path.join(dir, 'train', '*normal*'))
        num_to_take = min([len(i) for i in shards_names])
        print("Take {0} shards for each scene, totally about {1} frames.".format(num_to_take, num_to_take*512*4))
        for f in shards_names:
            shuffle(f)
            shards_taken = shards_taken + f[0:num_to_take]
        shuffle(shards_taken)
        return shards_taken
    else:
        filenames = []
        for dir in data_dir:
            filenames = filenames + glob.glob(os.path.join(dir, 'validation', 'validation*'))
        return filenames


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

    Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

    Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        # 'image/filename': tf.FixedLenFeature([], dtype=tf.string,
        #                                      default_value=''),
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'trajectory': tf.FixedLenFeature([], dtype=tf.string,
                                         default_value=''),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    return features['image/encoded'], features['trajectory']


def parse_record(raw_record):
    """Parses a record containing a training example of an image.

    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).

    Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

    Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    image_buffer, traj_buffer = _parse_example_proto(raw_record)

    # image = tf.decode_raw(image_buffer, tf.uint8)
    # image = tf.reshape(image, shape=[IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    # image = tf.cast(image, tf)
    image = tf.image.decode_png(image_buffer, channels=NUM_CHANNELS)
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    image = tf.cast(image, tf.float32)
    # normalize image
    image = image / 255.0
    # image = image - tf.convert_to_tensor(IMAGE_MEAN_ARRAY, dtype=tf.float32)

    traj = tf.decode_raw(traj_buffer, tf.int64)
    traj = tf.cast(traj, tf.float32)
    traj = tf.reshape(traj, shape=[FUTURE_TIME_STEP*2, ])
    # normailize traj
    # traj = traj / IMAGE_WIDTH

    return image, traj


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             dtype=tf.float32, datasets_num_private_threads=None,
             num_parallel_batches=1, parse_record_fn=parse_record):
    """Input function which provides batches for train or eval.

    Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    num_parallel_batches: Number of parallel batches for tf.data.
    parse_record_fn: Function to use for parsing the records.

    Returns:
    A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # dataset = dataset.map(parse_record)

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

    # iterator = dataset.make_one_shot_iterator()
    # image, traj = iterator.get_next()
    # return image, traj

    return dataset

