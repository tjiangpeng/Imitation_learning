from datetime import datetime
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from netTrain.GMM.net_model import ResNet50V2_gmm
from netTrain.GMM.custom_conf import log_likelihood_loss
from argoPrepare.load_tfrecord_argo import input_fn
# from utils_custom.load_tfrecord import input_fn
# from utils_custom.metrics import ADE_1S, FDE_1S, ADE_3S, FDE_3S
from hparms import *

NUM_EPOCHS = 10


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 40:
        lr *= 0.5e-3
    elif epoch > 30:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def main():
    keras.backend.clear_session()
    logtime = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "../../../logs/GMM/scalars/" + logtime

    # Tensorboard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # Save model weight
    checkpoint_callback = keras.callbacks.ModelCheckpoint("../../../logs/GMM/checkpoints/" + logtime + "weights{epoch:03d}.h5",
                                                          monitor='val_loss', save_best_only=True, mode='min',
                                                          save_weights_only=True)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0,
                                                   patience=5, min_lr=0.5e-6)

    callbacks = [tensorboard_callback, checkpoint_callback, lr_reducer, lr_scheduler]
    # callbacks = [lr_reducer, lr_scheduler]

    ####################################################################################################################
    # Dataset
    data_dir = ['../../../data/argo/forecasting/train/tf_record/']
    train_dataset = input_fn(is_training=True, data_dir=data_dir, batch_size=8, num_epochs=NUM_EPOCHS)

    data_dir = ['../../../data/argo/forecasting/val/tf_record/']
    valid_dataset = input_fn(is_training=False, data_dir=data_dir, batch_size=8, num_epochs=NUM_EPOCHS)
    ####################################################################################################################
    # Model
    model = ResNet50V2_gmm(weights=None,
                           input_img_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                           input_ptraj_shape=(PAST_TIME_STEP*2, ),
                           node_num=2048,
                           gmm_comp=NUM_GAUSSIAN_COMPONENT,
                           time_steps=NUM_TIME_SEQUENCE)

    model.compile(optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                  loss=log_likelihood_loss)

    history = model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=1, verbose=2, callbacks=callbacks,
                        validation_data=valid_dataset, validation_steps=8)  # 630

    with open(logdir + '/trainHistory.json', 'w') as f:
        history.history['lr'] = [float(i) for i in (history.history['lr'])]
        json.dump(history.history, f)


if __name__ == '__main__':
    main()
