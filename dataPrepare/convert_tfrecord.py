import os
import math
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

DATA_DIR = "../../data/"
# DATA_DIR = "sample/"

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'

TRAINING_SHARDS = 4
VALIDATION_SHARDS = 128

stopToken = False

# ==============================================================================
# -- Functuion -----------------------------------------------------------------
# ==============================================================================

def _check_or_create_dir(directory):
    """Check if drectory exists otherwise create it"""
    if not tf.io.gfile.exists(directory):
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

    # colorspace = b'BGR'
    # channels = 3
    # image_format = b'png'

    example = tf.train.Example(features=tf.train.Features(feature={
        # 'image/colorspace': _bytes_feature(colorspace),
        # 'image/channels': _int64_feature(channels),
        # 'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename).encode()),
        'image/encoded': _bytes_feature(image_buffer.tostring()),
        'trajectory': _bytes_feature(traj.tostring())
    }))

    return example


def _process_image(image_name, world):
    """Generate the final input image and ground truth"""
    global stopToken

    img = cv2.imread(image_name)
    ind = int(os.path.basename(image_name)[0:-4])

    traj = world.get_future_traj(ind)
    img = world.add_past_traj(img, ind)
    stopToken, img = world.generate_routing(ind, img)

    cv2.imshow("input image", img)

    cv2.waitKey(1)

    return img, traj


def _process_image_files_batch(world, output_file, image_names, log_names):
    """Processes and saves list of images as TFRecords"""
    global stopToken
    writer = tf.io.TFRecordWriter(output_file)

    for image_name, log_name in zip(image_names, log_names):
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
        chunk_logs = log_names[shard * chunksize : (shard + 1) * chunksize]
        output_file = os.path.join(
            output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))

        _process_image_files_batch(world, output_file, chunk_imgs, chunk_logs)
        tf.compat.v1.logging.info('Finished writing file: %s' % output_file)
        if stopToken:
            break


def convert_to_tf_records(raw_data_dir):
    """ Convert the raw dataset into TF-Record dumps"""

    # Glob all the training files
    training_images = sorted(tf.io.gfile.glob(
        os.path.join(raw_data_dir, TRAINING_DIRECTORY, 'img', '*.png')))

    training_logs = sorted(tf.io.gfile.glob(
        os.path.join(raw_data_dir, TRAINING_DIRECTORY, 'txt', '*.txt')))

    ARGS.sequence_ind = [int(os.path.basename(training_images[0])[0:-4]),
                         int(os.path.basename(training_images[-1])[0:-4])]

    # Create training data
    tf.compat.v1.logging.info('Processing the training data.')
    _process_dataset(
        training_images, training_logs,
        os.path.join(raw_data_dir, TRAINING_DIRECTORY), TRAINING_DIRECTORY, TRAINING_SHARDS)


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
    convert_to_tf_records(DATA_DIR)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()