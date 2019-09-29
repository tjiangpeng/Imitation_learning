import time

import numpy as np
from hparms import *


class HardSampleReservoir:
    def __init__(self):
        self.input_data = None
        self.output_data = None
        self.size = 0
        self.input_buffer = None
        self.output_buffer = None

    def append(self, dt, gt, ind: list):
        start_time = time.time()
        if ind:
            if self.size:
                self.input_data['input_1'] = np.concatenate((dt['input_1'][ind, :, :, :], self.input_data['input_1']), axis=0)
                self.input_data['input_2'] = np.concatenate((dt['input_2'][ind, :], self.input_data['input_2']), axis=0)
                self.output_data = np.concatenate((gt[ind, :], self.output_data), axis=0)
            else:
                self.input_data = {'input_1': dt['input_1'][ind, :, :, :],
                                   'input_2': dt['input_2'][ind, :]}
                self.output_data = gt[ind, :]

            if self.output_data.shape[0] > RESERVOIR_MAX_SIZE:
                self.input_data['input_1'] = self.input_data['input_1'][0:RESERVOIR_MAX_SIZE, :, :, :]
                self.input_data['input_2'] = self.input_data['input_2'][0:RESERVOIR_MAX_SIZE, :]
                self.output_data = self.output_data[0:RESERVOIR_MAX_SIZE, :]

            self.size = self.output_data.shape[0]
#        print("----------------------------")
#        print("Reservoir size")
#        print(self.size)
        print(f"--- Append time: {int(time.time() - start_time)} second ---")

    def pop(self):
        start_time = time.time()
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

        print(f"--- Pop time: {int(time.time() - start_time)} second ---")
        return dt, gt

    def push_to_buffer(self, dt, gt):
        start_time = time.time()

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

#        print("----------------------------")
#        print("Buffer size")
#        print(self.output_buffer.shape[0])
#        print("Reservoir size")
#        print(self.size)

        print(f"--- Push buffer time: {int(time.time() - start_time)} second ---")
