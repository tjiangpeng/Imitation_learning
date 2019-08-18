import os
import math
from random import shuffle
# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import cv2
import argparse
import tensorflow as tf
from dataPrepare.raw_data_process import World, load_parms

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================

DATA_DIR = "../../data/2019_08_07/"
# DATA_DIR = "sample/"

IS_TRAINING = False
TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'

TRAINING_SHARDS = 4
VALIDATION_SHARDS = 1

stopToken = False

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


def _convert_to_example(filename, image_buffer, traj):
    """Build an Example proto for an example"""

    cv2.imwrite('test.png', image_buffer)
    img_bytes = open('test.png', 'rb').read()

    # colorspace = b'BGR'
    # channels = 3
    # image_format = b'png'

    example = tf.train.Example(features=tf.train.Features(feature={
        # 'image/colorspace': _bytes_feature(colorspace),
        # 'image/channels': _int64_feature(channels),
        # 'image/format': _bytes_feature(image_format),
        # 'image/filename': _bytes_feature(os.path.basename(filename).encode()),
        'image/encoded': _bytes_feature(img_bytes),
        'trajectory': _bytes_feature(traj.tostring())
    }))

    return example


def _process_image(image_name, world):
    """Generate the final input image and ground truth"""
    global stopToken

    img = cv2.imread(image_name)
    ind = int(os.path.basename(image_name)[0:-4])

    traj = world.get_future_traj(ind)
    # img = world.add_past_traj(img, ind, 1)
    stopToken, img = world.generate_routing(ind, img)
    img = world.add_past_corners(img, ind)

    cv2.imshow("input image", img)

    cv2.waitKey(1)

    return img, traj


def _process_image_files_batch(world, output_file, image_names):
    """Processes and saves list of images as TFRecords"""
    global stopToken, ARGS
    writer = tf.io.TFRecordWriter(output_file)

    for image_name in image_names:
        ind = int(os.path.basename(image_name)[0:-4])
        if ind in ARGS.outliers:
            print("frame " + str(ind) + " is removed!")
            continue

        image_buffer, traj = _process_image(image_name, world)
        if stopToken:
            break
        example = _convert_to_example(image_name, image_buffer, traj)

        writer.write(example.SerializeToString())

    writer.close()


def _process_dataset(image_names, log_names, output_directory, prefix, num_shards):
    """Processes and saves list of images as TFRecords."""
    global stopToken
    # Create carla world
    world = World(ARGS, log_names, timeout=2.0)

    # Process
    _check_or_create_dir(output_directory)
    chunksize = int(math.ceil(len(image_names) / num_shards))

    for shard in range(num_shards):
        chunk_imgs = image_names[shard * chunksize : (shard + 1) * chunksize]
        output_file = os.path.join(
            output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))

        _process_image_files_batch(world, output_file, chunk_imgs)
        tf.logging.info('Finished writing file: %s' % output_file)
        if stopToken:
            break


def load_outlier_ind(dir):
    outliers = []
    try:
        f = open(dir, "r")
        fl = f.readlines()
        for line in fl:
            start = int(line.split(",")[0])
            end = int(line.split(",")[1])
            outliers = outliers + [i for i in range(start, end+1)]
    except IOError:
        print("the outlier file doesn't exist, which means no outlier!")

    return outliers


def convert_to_tf_records(is_training=True, raw_data_dir=None):
    """ Convert the raw dataset into TF-Record dumps"""
    global ARGS
    if is_training:
        directory = TRAINING_DIRECTORY
        shards = TRAINING_SHARDS
    else:
        directory = VALIDATION_DIRECTORY
        shards = VALIDATION_SHARDS

    ARGS.outliers = load_outlier_ind(os.path.join(raw_data_dir, directory, "outliers.txt"))

    # Glob all the files
    images = sorted(tf.gfile.Glob(
        os.path.join(raw_data_dir, directory, 'img', '*.png')))

    logs = sorted(tf.gfile.Glob(
        os.path.join(raw_data_dir, directory, 'txt', '*.txt')))

    ARGS.sequence_ind = [int(os.path.basename(images[0])[0:-4]),
                         int(os.path.basename(images[-1])[0:-4])]

    if is_training:
        shuffle(images)
    # Create training data
    tf.logging.info('Processing the data.')
    _process_dataset(
        images, logs,
        os.path.join(raw_data_dir, directory), directory, shards)


def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(
        description='Data preparation')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--map',
        metavar='TOWN',
        default=None,
        help='start a new episode at the given TOWN')

    global ARGS
    ARGS = argparser.parse_args()
    ARGS.description = argparser.description
    ARGS = load_parms(DATA_DIR + 'parms.txt', ARGS)

    # Convert the raw data into tf-records
    convert_to_tf_records(is_training=IS_TRAINING, raw_data_dir=DATA_DIR)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()