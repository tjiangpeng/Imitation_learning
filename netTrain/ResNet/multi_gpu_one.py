import os

import tensorflow as tf
from tensorflow import keras
from netTrain.ResNet.net_model import ResNet50V2, ResNet50V2_fc
from hparms import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"  # specify which GPU(s) to be used


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    model = ResNet50V2(include_top=True, weights=None,
                       input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                       classes=NUM_TIME_SEQUENCE*2)

    # model = ResNet50V2_fc(weights=None,
    #                       input_img_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
    #                       input_ptraj_shape=(PAST_TIME_STEP*2, ),
    #                       node_num=2048,
    #                       classes=NUM_TIME_SEQUENCE*2)

    model = keras.utils.multi_gpu_model(model, gpus=4)
    model.load_weights('../../../logs/ResNet/checkpoints/20190919-101758weights024.h5')

    new_model = model.layers[-2]
    new_model.save('new_model.h5')


if __name__ == '__main__':
    main()
