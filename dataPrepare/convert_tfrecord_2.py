"""
generate small classified shards of tfrecord files
"""
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

DATA_DIR = "../../data/2019_08_12/"
# DATA_DIR = "sample/"
IS_TRAINING = True

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'
FRAME_IN_SHARD = 512

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

    stopToken, img = world.generate_routing(ind, img)
    if stopToken:
        return 0, 0, 0
    traj = world.get_future_traj(ind)
    img = world.add_past_corners(img, ind)

    cv2.imshow("input image", img)
    cv2.waitKey(1)

    vari_rowInd = traj[:, 0].var()
    # print(vari_rowInd)
    if vari_rowInd >= 2:
        return img, traj, "bend_road"
    if world.log[ind]['traffic_light'] == 'Red' or world.log[ind]['traffic_light'] == 'Yellow':
        return img, traj, "red_light"
    if (world.log[ind]['traffic_light'] == 'Green' or world.log[ind]['traffic_light'] == 'None') \
            and world.log[ind]['speed'] < (world.log[ind]['speed_limit'] - 15):
        return img, traj, "front_vehicle"
    return img, traj, "normal"


def _process_dataset(image_names, log_names, output_directory, prefix):
    """Processes and saves list of images as TFRecords."""
    global stopToken, ARGS
    # Create carla world
    world = World(ARGS, log_names, timeout=2.0)

    # Process
    _check_or_create_dir(output_directory)

    shards = [1, 1, 1, 1]
    writer_bend_road = tf.io.TFRecordWriter(os.path.join(output_directory, '%s-bend-road-%.3d' % (prefix, shards[0])))
    writer_red_light = tf.io.TFRecordWriter(os.path.join(output_directory, '%s-red-light-%.3d' % (prefix, shards[1])))
    writer_front_vehicle = tf.io.TFRecordWriter(os.path.join(output_directory, '%s-front-vehicle-%.3d' % (prefix, shards[2])))
    writer_normal = tf.io.TFRecordWriter(os.path.join(output_directory, '%s-normal-%.3d' % (prefix, shards[3])))

    num_frame = [0, 0, 0, 0]
    for image_name in image_names:
        ind = int(os.path.basename(image_name)[0:-4])
        if ind in ARGS.outliers:
            print("frame " + str(ind) + " is removed!")
            continue

        image_buffer, traj, mode = _process_image(image_name, world)
        if stopToken:
            continue
        example = _convert_to_example(image_name, image_buffer, traj)

        if mode == "bend_road":
            writer_bend_road.write(example.SerializeToString())
            num_frame[0] += 1
            if num_frame[0] % FRAME_IN_SHARD == 0:
                shards[0] += 1
                writer_bend_road.close()
                writer_bend_road = tf.io.TFRecordWriter(
                    os.path.join(output_directory, '%s-bend-road-%.3d' % (prefix, shards[0])))
        elif mode == "red_light":
            writer_red_light.write(example.SerializeToString())
            num_frame[1] += 1
            if num_frame[1] % FRAME_IN_SHARD == 0:
                shards[1] += 1
                writer_red_light.close()
                writer_red_light = tf.io.TFRecordWriter(
                    os.path.join(output_directory, '%s-red-light-%.3d' % (prefix, shards[1])))
        elif mode == "front_vehicle":
            writer_front_vehicle.write(example.SerializeToString())
            num_frame[2] += 1
            if num_frame[2] % FRAME_IN_SHARD == 0:
                shards[2] += 1
                writer_front_vehicle.close()
                writer_front_vehicle = tf.io.TFRecordWriter(
                    os.path.join(output_directory, '%s-front-vehicle-%.3d' % (prefix, shards[2])))
        else:
            writer_normal.write(example.SerializeToString())
            num_frame[3] += 1
            if num_frame[3] % FRAME_IN_SHARD == 0:
                shards[3] += 1
                writer_normal.close()
                writer_normal = tf.io.TFRecordWriter(
                    os.path.join(output_directory, '%s-normal-%.3d' % (prefix, shards[3])))

    writer_bend_road.close()
    writer_red_light.close()
    writer_front_vehicle.close()
    writer_normal.close()

    f = open(os.path.join(output_directory, 'frame-number.txt'), "w")
    f.write("bend road frame: {0}, shards: {1}\n".format(num_frame[0], shards[0]))
    f.write("red light frame: {0}, shards: {1}\n".format(num_frame[1], shards[1]))
    f.write("front vehicle frame: {0}, shards: {1}\n".format(num_frame[2], shards[2]))
    f.write("normal frame: {0}, shards: {1}\n".format(num_frame[3], shards[3]))

    f.close()


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


def convert_to_tf_records(is_training, raw_data_dir):
    """ Convert the raw dataset into TF-Record dumps"""
    global ARGS

    if is_training:
        directory = TRAINING_DIRECTORY
    else:
        directory = VALIDATION_DIRECTORY

    ARGS.outliers = load_outlier_ind(os.path.join(raw_data_dir, directory, "outliers.txt"))

    # Glob all the files
    images = sorted(tf.gfile.Glob(
        os.path.join(raw_data_dir, directory, 'img', '*.png')))

    logs = sorted(tf.gfile.Glob(
        os.path.join(raw_data_dir, directory, 'txt', '*.txt')))

    ARGS.sequence_ind = [int(os.path.basename(images[0])[0:-4]),
                         int(os.path.basename(images[-1])[0:-4])]

    if is_training:
        # shuffle the images
        shuffle(images)
    # Create training data
    tf.logging.info('Processing the data.')
    _process_dataset(
        images, logs,
        os.path.join(raw_data_dir, directory), directory)


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
    convert_to_tf_records(IS_TRAINING, DATA_DIR)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()