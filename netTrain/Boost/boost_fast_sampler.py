import time

import numpy as np
from hparms import *


class HardSampleReservoir:
    def __init__(self):
        self.input_data = {'input_1': np.zeros([RESERVOIR_MAX_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS], dtype=np.float32),
                           'input_2': np.zeros([RESERVOIR_MAX_SIZE, PAST_TIME_STEP*2], dtype=np.float32)}
        self.output_data = np.zeros([RESERVOIR_MAX_SIZE, FUTURE_TIME_STEP*2], dtype=np.float32)
        self.data_size = 0
        self.head_ptr = 0

        self.input_buffer = {'input_1': np.zeros([(HARD_SAMPLE_RATIO+1)*BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS], dtype=np.float32),
                             'input_2': np.zeros([(HARD_SAMPLE_RATIO+1)*BATCH_SIZE, PAST_TIME_STEP*2], dtype=np.float32)}
        self.output_buffer = np.zeros([(HARD_SAMPLE_RATIO+1)*BATCH_SIZE, FUTURE_TIME_STEP*2], dtype=np.float32)
        self.buffer_size = 0

    def append(self, dt, gt, ind: list):
        # start_time = time.time()
        if ind:
            append_size = len(ind)
            tail_ptr = (self.head_ptr + self.data_size) % RESERVOIR_MAX_SIZE

            if tail_ptr + append_size >= RESERVOIR_MAX_SIZE:
                range_ind = np.arange(tail_ptr, RESERVOIR_MAX_SIZE).tolist() + \
                            np.arange(0, tail_ptr+append_size-RESERVOIR_MAX_SIZE).tolist()
            else:
                range_ind = np.arange(tail_ptr, tail_ptr+append_size).tolist()

            self.input_data['input_1'][range_ind, :, :, :] = dt['input_1'][ind, :, :, :]
            self.input_data['input_2'][range_ind, :] = dt['input_2'][ind, :]
            self.output_data[range_ind, :] = gt[ind, :]

            if self.data_size + append_size > RESERVOIR_MAX_SIZE:
                self.data_size = RESERVOIR_MAX_SIZE
                self.head_ptr = (tail_ptr + append_size) % RESERVOIR_MAX_SIZE
            else:
                self.data_size = self.data_size + append_size

        # print("----------------------------")
        # print(f"ind: {ind}")
        # print(f"Reservoir size: {self.data_size}")
        # print(f"head: {self.head_ptr}")
        # print(f"--- Append time: {(time.time() - start_time)} second ---")
        # print(self.output_data)

    def pop(self):
        # start_time = time.time()

        avail_size = min(self.buffer_size, BATCH_SIZE)
        dt = {'input_1': self.input_buffer['input_1'][self.buffer_size-avail_size:self.buffer_size, :, :, :],
              'input_2': self.input_buffer['input_2'][self.buffer_size-avail_size:self.buffer_size, :]}
        gt = self.output_buffer[self.buffer_size-avail_size:self.buffer_size, :]

        self.buffer_size = self.buffer_size - avail_size

        # print("----------------------------")
        # print("Pop size")
        # print(gt.shape[0])
        # print("Buffer size")
        # print(self.output_buffer.shape[0])

        # print(f"--- Pop time: {(time.time() - start_time)} second ---")
        return dt, gt

    def push_to_buffer(self, dt, gt):
        # start_time = time.time()

        self.input_buffer['input_1'][0:BATCH_SIZE, :, :, :] = dt['input_1']
        self.input_buffer['input_2'][0:BATCH_SIZE, :] = dt['input_2']
        self.output_buffer[0:BATCH_SIZE] = gt
        self.buffer_size = BATCH_SIZE
        if self.data_size:
            avail_size = min(self.data_size, BATCH_SIZE*HARD_SAMPLE_RATIO)

            if self.head_ptr + avail_size >= RESERVOIR_MAX_SIZE:
                range_ind = np.arange(self.head_ptr, RESERVOIR_MAX_SIZE).tolist() + \
                            np.arange(0, self.head_ptr + avail_size - RESERVOIR_MAX_SIZE).tolist()
                self.head_ptr = self.head_ptr + avail_size - RESERVOIR_MAX_SIZE
            else:
                range_ind = np.arange(self.head_ptr, self.head_ptr+avail_size).tolist()
                self.head_ptr = self.head_ptr + avail_size

            self.input_buffer['input_1'][BATCH_SIZE:BATCH_SIZE+avail_size, :, :, :] = \
                self.input_data['input_1'][range_ind, :, :, :]
            self.input_buffer['input_2'][BATCH_SIZE:BATCH_SIZE+avail_size, :] = self.input_data['input_2'][range_ind, :]
            self.output_buffer[BATCH_SIZE:BATCH_SIZE+avail_size, :] = self.output_data[range_ind, :]

            self.buffer_size = BATCH_SIZE + avail_size
            self.data_size = self.data_size - avail_size

            # Shuffle the buffer
            order = np.arange(self.buffer_size)
            np.random.shuffle(order)
            order = order.tolist()
            self.input_buffer['input_1'][0:self.buffer_size, :, :, :] = self.input_buffer['input_1'][order, :, :, :]
            self.input_buffer['input_2'][0:self.buffer_size, :] = self.input_buffer['input_2'][order, :]
            self.output_buffer[0:self.buffer_size, :] = self.output_buffer[order, :]

#        print("----------------------------")
#        print("Buffer size")
#        print(self.output_buffer.shape[0])
#        print("Reservoir size")
#        print(self.size)

        # print(f"--- Push buffer time: {(time.time() - start_time)} second ---")
        # print(f"head: {self.head_ptr}")
        # print(self.output_buffer)
