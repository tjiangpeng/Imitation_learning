import os
import tensorflow as tf
import keras
import cv2
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from netTrain.VggNet.net_model import VGG16
from official.resnet import resnet_run_loop
from keras import backend as K

###############################################################################
# Constant
###############################################################################
IMAGE_HEIGHT = 704
IMAGE_WIDTH = 704
NUM_CHANNELS = 3
NUM_TIME_SEQUENCE = 10
NUM_HIDDEN_NODES = 4096

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 4
_NUM_VALIDATION_FILES = 128
_SHUFFLE_BUFFER = 10000


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'train-%.5d-of-%.5d' % (i, _NUM_TRAIN_FILES))
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation-%.5d-of-%.5d' % (i, _NUM_VALIDATION_FILES))
            for i in range(_NUM_VALIDATION_FILES)]


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
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
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

    image = tf.decode_raw(image_buffer, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    # image = tf.cast(image, tf)

    traj = tf.decode_raw(traj_buffer, tf.int64)
    traj = tf.cast(traj, tf.float32)
    traj = tf.reshape(traj, shape=[NUM_TIME_SEQUENCE*2, ])

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
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_record)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    image, traj = iterator.get_next()

    return image, traj


def main():
    # K.tensorflow_backend._get_available_gpus()

    image, traj = input_fn(is_training=True, data_dir='../../../data/2019_07_25/train/',
                       batch_size=1, num_epochs=10)

    # sess = tf.Session()
    # iterator = dataset.make_one_shot_iterator()
    # image_batch, traj_batch = iterator.get_next()
    #
    # for i in range(100):
    #     im, la = sess.run([image_batch, traj_batch])
    #     cv2.imshow('image', im[0])
    #     cv2.waitKey(10)
    #     print(la)
    #
    #     print(i)
    # sess.close()

    vgg_model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(tensor=image, shape=(704, 704, 3)),
                      input_shape=(704, 704, 3), pooling='max')
    last_layer = vgg_model.get_layer('block5_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(NUM_HIDDEN_NODES, activation='relu', name='fc1')(x)
    x = Dense(NUM_HIDDEN_NODES, activation='relu', name='fc2')(x)
    out = Dense(NUM_TIME_SEQUENCE*2, activation='relu', name='fc3')(x)
    model = Model(vgg_model.input, out)

    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
                  loss='mse',
                  metrics=['mae'],
                  target_tensors=[traj])
    model.fit(epochs=10, steps_per_epoch=1, verbose=2)
    pass


if __name__ == '__main__':
    main()