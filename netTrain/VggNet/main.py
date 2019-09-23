from datetime import datetime
import json
import numpy as np
from tensorflow import keras
from netTrain.VggNet.net_model import VGG16_FL
from utils.load_tfrecord import input_fn
from hparms import *

NUM_HIDDEN_NODES = 2500
NUM_EPOCHS = 20

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-7
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
    logdir = "logs/scalars/" + logtime

    # Tensorboard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # Save model weight
    checkpoint_callback = keras.callbacks.ModelCheckpoint("logs/checkpoints/" + logtime + "weights{epoch:04d}.h5",
                                                          monitor='val_loss', save_best_only=True, mode='min',
                                                          save_weights_only=True)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='mean_absolute_error', factor=np.sqrt(0.1), cooldown=0,
                                                   patience=5, min_lr=0.5e-6)

    callbacks = [tensorboard_callback, checkpoint_callback, lr_reducer, lr_scheduler]

    ####################################################################################################################
    # Dataset
    data_dir = ['../../../data/2019_08_07/', '../../../data/2019_08_10/',
                '../../../data/2019_08_12/', '../../../data/2019_08_12_2/']
    train_dataset = input_fn(is_training=True, data_dir=data_dir, batch_size=8, num_epochs=NUM_EPOCHS)

    data_dir = ['../../../data/2019_08_07/']#, '../../../data/2019_08_14/']
    valid_dataset = input_fn(is_training=False, data_dir=data_dir, batch_size=8, num_epochs=NUM_EPOCHS)
    ####################################################################################################################
    # Model
    model = VGG16_FL(num_node=NUM_HIDDEN_NODES, num_time_sequence=FUTURE_TIME_STEP,
                     weight='logs/checkpoints/20190815-171720weights0062.h5') #'logs/checkpoints/20190815-084234weights0010.h5'

    model.compile(optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                  loss='mse',
                  metrics=['mae'])

    history = model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=3000, verbose=2, callbacks=callbacks,
                        validation_data=valid_dataset, validation_steps=620) # 620

    with open(logdir + '/trainHistory.json', 'w') as f:
        history.history['lr'] = [float(i) for i in (history.history['lr'])]
        json.dump(history.history, f)


if __name__ == '__main__':
    main()
