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
from utils_custom.utils_argo import ADE_1S, FDE_1S, ADE_3S, FDE_3S, ADE_FDE_loss, ADE_3S_array
from hparms import *

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # specify which GPU(s) to be used

NUM_EPOCHS = 10
STEPS_PER_EPOCH = 100
BATCH_SIZE = 8
HARD_SAMPLE_RATIO = 3
RESERVOIR_MAX_SIZE = BATCH_SIZE * 50

NORMAL_TRAIN_EPOCH = 0
LOSS_THRESHOLD = 1.0


class HardSampleReservoir:
    def __init__(self):
        self.input_data = None
        self.output_data = None
        self.size = 0
        self.input_buffer = None
        self.output_buffer = None

    def append(self, dt, gt, ind: list):
        if self.size:
            self.input_data['input_1'] = np.concatenate((self.input_data['input_1'], dt['input_1'][ind, :, :, :]), axis=0)
            self.input_data['input_2'] = np.concatenate((self.input_data['input_2'], dt['input_2'][ind, :]), axis=0)
            self.output_data = np.concatenate((self.output_data, gt[ind, :]), axis=0)
        else:
            self.input_data = {'input_1': dt['input_1'][ind, :, :, :],
                               'input_2': dt['input_2'][ind, :]}
            self.output_data = gt[ind, :]

        if self.output_data.shape[0] > RESERVOIR_MAX_SIZE:
            self.input_data['input_1'] = self.input_data['input_1'][0:RESERVOIR_MAX_SIZE, :, :, :]
            self.input_data['input_2'] = self.input_data['input_2'][0:RESERVOIR_MAX_SIZE, :]
            self.output_data = self.output_data[0:RESERVOIR_MAX_SIZE, :]

        self.size = self.output_data.shape[0]
        print("----------------------------")
        print("Reservoir size")
        print(self.size)

    def pop(self):
        dt = {'input_1': self.input_buffer['input_1'][-BATCH_SIZE::, :, :, :],
              'input_2': self.input_buffer['input_2'][-BATCH_SIZE::, :]}
        gt = self.output_buffer[-BATCH_SIZE::, :]

        self.input_buffer['input_1'] = self.input_buffer['input_1'][0:-BATCH_SIZE, :, :, :]
        self.input_buffer['input_2'] = self.input_buffer['input_2'][0:-BATCH_SIZE, :]
        self.output_buffer = self.output_buffer[0:-BATCH_SIZE, :]

        # print("----------------------------")
        # print("Pop size")
        # print(gt.shape[0])
        # print("Buffer size")
        # print(self.output_buffer.shape[0])
        return dt, gt

    def push_to_buffer(self, dt, gt):
        self.input_buffer = dt
        self.output_buffer = gt
        if self.size:
            # Concatenate the normal data and hard samples as buffer
            self.input_buffer['input_1'] = np.concatenate(
                (self.input_buffer['input_1'],
                 self.input_data['input_1'][-BATCH_SIZE*HARD_SAMPLE_RATIO::, :, :, :]), axis=0
            )
            self.input_buffer['input_2'] = np.concatenate(
                (self.input_buffer['input_2'],
                 self.input_data['input_2'][-BATCH_SIZE*HARD_SAMPLE_RATIO::, :]), axis=0
            )
            self.output_buffer = np.concatenate(
                (self.output_buffer, self.output_data[-BATCH_SIZE*HARD_SAMPLE_RATIO::, :]), axis=0
            )
            # Remove the concatenated hard samples from data reservoir
            self.input_data['input_1'] = self.input_data['input_1'][0:-BATCH_SIZE * HARD_SAMPLE_RATIO, :, :, :]
            self.input_data['input_2'] = self.input_data['input_2'][0:-BATCH_SIZE * HARD_SAMPLE_RATIO, :]
            self.output_data = self.output_data[0:-BATCH_SIZE * HARD_SAMPLE_RATIO, :]
            self.size = self.output_data.shape[0]

            # Shuffle the buffer
            order = np.arange(self.output_buffer.shape[0])
            np.random.shuffle(order)
            order = order.tolist()
            self.input_buffer['input_1'] = self.input_buffer['input_1'][order, :, :, :]
            self.input_buffer['input_2'] = self.input_buffer['input_2'][order, :]
            self.output_buffer = self.output_buffer[order, :]

        # print("----------------------------")
        # print("Buffer size")
        # print(self.output_buffer.shape[0])
        # print("Reservoir size")
        # print(self.size)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-6
    if epoch > 30:
        lr *= 0.5e-3
    elif epoch > 25:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 5:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def main():
    keras.backend.clear_session()
    sess = tf.Session()

    logtime = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "../../../logs/ResNet/scalars/" + logtime

    # Tensorboard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # Save model weight
    checkpoint_callback = keras.callbacks.ModelCheckpoint("../../../logs/ResNet/checkpoints/" + logtime + "weights{epoch:03d}.h5",
                                                          monitor='val_loss', save_best_only=True, mode='min',
                                                          save_weights_only=True)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    callbacks = [tensorboard_callback, checkpoint_callback, lr_scheduler]

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
    model = ResNet50V2(include_top=True, weights=None,
                       input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                       classes=FUTURE_TIME_STEP*2)
    # model = ResNet50V2_fc(weights=None,
    #                       input_img_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
    #                       input_ptraj_shape=(PAST_TIME_STEP*2, ),
    #                       node_num=2048,
    #                       classes=NUM_TIME_SEQUENCE*2)

    # model = keras.utils.multi_gpu_model(model, gpus=4)
    # model.load_weights('../../../logs/ResNet/checkpoints/20190925-201705weights028.h5')

    model.compile(optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                  loss=ADE_3S,
                  metrics=[ADE_1S, FDE_1S, ADE_3S, FDE_3S])

    # ==============================================================================
    # -- Training -----------------------------------------------------------------
    # ==============================================================================
    reservior = HardSampleReservoir()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}/{NUM_EPOCHS}")
        model.optimizer.lr = lr_schedule(epoch)
        start_time = time.time()

        for step in range(STEPS_PER_EPOCH):
            dt, gt = sess.run([input_batch, gt_batch])

            re = [0, 0]  # result of training loss and metrics
            if epoch < NORMAL_TRAIN_EPOCH:
                out = model.train_on_batch(dt, gt)
                re = [re[0] + out, re[1] + gt.shape[0]]
            else:
                reservior.push_to_buffer(dt, gt)
                for i in range(1+HARD_SAMPLE_RATIO):
                    dt, gt = reservior.pop()
                    if gt.size:
                        out = model.train_on_batch(dt, gt)
                        re = [re[0] + out, re[1] + gt.shape[0]]

                        # Find the hard samples
                        y_pred = model.predict(dt)
                        loss = ADE_3S_array(gt, y_pred)
                        ind = np.where(loss > LOSS_THRESHOLD)[0]

                        # Push the hard samples into reservoir
                        reservior.append(dt, gt, ind.tolist())

        # TODO: Print the training result
        print(f"--- Training time: {time.time()-start_time} ---")

        # TODO: Evaluate the validation dataset

        # TODO: Save model

    # history = model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=1600, verbose=2, callbacks=callbacks,
    #                     validation_data=valid_dataset, validation_steps=2507)  # 40127
    #
    # with open(logdir + '/trainHistory.json', 'w') as f:
    #     history.history['lr'] = [float(i) for i in (history.history['lr'])]
    #     json.dump(history.history, f)


if __name__ == '__main__':
    main()
