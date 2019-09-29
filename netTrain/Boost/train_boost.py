import os
from datetime import datetime
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from netTrain.ResNet.net_model import ResNet50V2, ResNet50V2_fc
# from argoPrepare.load_tfrecord_argo import input_fn
from argoData.load_tfrecord_argo import input_fn
# from utils_custom.load_tfrecord import input_fn
from utils_custom.utils_argo import ADE_1S, FDE_1S, ADE_2S, FDE_2S, ADE_3S, FDE_3S, ADE_FDE_loss, metrics_array
from netTrain.Boost.boost_sampler import HardSampleReservoir
from hparms import *

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # specify which GPU(s) to be used


# ==============================================================================
# -- Function -----------------------------------------------------------------
# ==============================================================================


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 30:
        lr *= 0.5e-3
    elif epoch > 25:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    logtime = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = "../../../logs/Boost/scalars/" + logtime

    # # Tensorboard
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # ==============================================================================
    # -- Dataset -----------------------------------------------------------------
    # ==============================================================================
    data_dir = ['../../../data/argo/forecasting/train/tf_record_4_channel/']
    train_dataset = input_fn(is_training=True, data_dir=data_dir, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
    iterator = train_dataset.make_one_shot_iterator()
    input_batch, gt_batch = iterator.get_next()

    data_dir = ['../../../data/argo/forecasting/val/tf_record_4_channel/']
    valid_dataset = input_fn(is_training=False, data_dir=data_dir, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)

    # ==============================================================================
    # -- Model -----------------------------------------------------------------
    # ==============================================================================
    # model = ResNet50V2(include_top=True, weights=None,
    #                    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
    #                    classes=FUTURE_TIME_STEP*2)
    model = ResNet50V2_fc(weights=None,
                          input_img_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                          input_ptraj_shape=(PAST_TIME_STEP*2, ),
                          node_num=2048,
                          classes=FUTURE_TIME_STEP*2)

    # model = keras.utils.multi_gpu_model(model, gpus=4)
    model.load_weights('../../../logs/ResNet/checkpoints/20190926-115346weights018.h5')

    model.compile(optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                  loss=ADE_FDE_loss,
                  metrics=[ADE_1S, ADE_2S, ADE_3S, FDE_1S, FDE_2S, FDE_3S])

    # ==============================================================================
    # -- Training -----------------------------------------------------------------
    # ==============================================================================
    reservior = HardSampleReservoir()
    min_loss = 10000
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}/{NUM_EPOCHS}")
        model.optimizer.lr = lr_schedule(epoch)
        start_time = time.time()

        re = [0, 0]  # result of training loss and metrics
        for step in range(STEPS_PER_EPOCH):
            dt, gt = sess.run([input_batch, gt_batch])

            if epoch < NORMAL_TRAIN_EPOCH:
                out = model.train_on_batch(dt, gt)
                re = [re[0] + np.array(out)*gt.shape[0], re[1] + gt.shape[0]]
            else:
                reservior.push_to_buffer(dt, gt)
                for i in range(1+HARD_SAMPLE_RATIO):  # n*Hard samples + 1*normal samples
                    dt, gt = reservior.pop()
                    if gt.size:
                        out = model.train_on_batch(dt, gt)
                        re = [re[0] + np.array(out)*gt.shape[0], re[1] + gt.shape[0]]

                        # Find the hard samples
                        y_pred = model.predict(dt)
                        loss = metrics_array(gt, y_pred)
                        ind = []
                        for ii in range(loss.shape[0]):
                            if loss[ii, 0] > valid_scores[1] or loss[ii, 1] > valid_scores[2] \
                                    or loss[ii, 2] > valid_scores[3] or loss[ii, 3] > valid_scores[4] \
                                    or loss[ii, 4] > valid_scores[5] or loss[ii, 5] > valid_scores[6]:
                                ind.append(ii)

                        # Push the hard samples into reservoir
                        reservior.append(dt, gt, ind)

        # Evaluate the validation dataset
        valid_scores = model.evaluate(valid_dataset, verbose=0, steps=2507)

        # Print the training result
        print(f"--- Time: {int(time.time()-start_time)} second ---")
        for i, score in enumerate(re[0]):
            print(f"--{model.metrics_names[i]}: {score/re[1]}", end=' ')
        print("")

        for i, score in enumerate(valid_scores):
            if i == 0:
                print(f"--valid_loss: {valid_scores[0]}", end=' ')
            else:
                print(f"--{model.metrics_names[i]}: {score}", end=' ')
        print("")

        # Save the best model according to validation result
        if valid_scores[0] < min_loss:
            if epoch >= NORMAL_TRAIN_EPOCH:
                model.save_weights(f"../../../logs/Boost/checkpoints/{logtime}weights{epoch:03d}.h5")
            min_loss = valid_scores[0]

        print(f"reservoir size: {reservior.size}")


if __name__ == '__main__':
    main()
