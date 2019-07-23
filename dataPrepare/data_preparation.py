import sys
import os
import glob
import cv2

# DATA_DIR = "../../data/"
DATA_DIR = "sample/"


def load_parms(dir):
    f = open(dir, "r")
    fl = f.readlines()

    parms = dict()
    parms['pixels_per_meter'] = int(fl[1])
    parms['pixels_ahead_hero'] = int(fl[3])
    parms['past_time_interval'] = int(fl[5])
    parms['image_res'] = [int(i) for i in fl[7].split(',')]
    parms['sequence_ind'] = [int(i) for i in fl[9].split(',')]

    return parms


def load_txt(dir, time):
    f = open(dir, "r")
    fl = f.readlines()

    data = dict()
    data['time'] = float(fl[1])
    data['hero_global_pos'] = [float(i) for i in fl[3].split(",")]
    data['road_id'] = int(fl[5])
    data['speed_limit'] = float(fl[7])
    data['speed'] = float(fl[9])
    data['translation_offset'] = [float(i) for i in fl[11].split(",")]
    data['angle'] = float(fl[13])

    data['hero_past_traj'] = []
    for pos in fl[15:(15+time-1)]:
        data['hero_past_traj'].append([int(i) for i in pos.split(",")])

    data['actor_past_traj'] = []
    num = int((len(fl) - 26) / (time+1))
    for ii in range(num):
        past_pos = []
        for pos in fl[(27+ii*(time+1)):(27+time-1+ii*(time+1))]:
            past_pos.append([int(i) for i in pos.split(",")])
        data['actor_past_traj'].append(past_pos)

    return data


def add_past_traj(hd_map, log, time, res):
    decrease = int(255 / (time + 1)) - 1
    color = [255, 255-decrease, 255-decrease]
    for past_pos in log['hero_past_traj']:
        if past_pos[0] >= 0 and past_pos[0] <= res[0] and past_pos[1] >= 0 and past_pos[1] <= res[1]:
            # print(hd_map[past_pos[0]-1:past_pos[0]+1, past_pos[1]-1:past_pos[1]+1])
            hd_map[past_pos[1]-1:past_pos[1]+2][past_pos[0]-1:past_pos[0]+2] = color
        color[1:2] = [color[1] - decrease, color[2] - decrease]

    return hd_map


def main():
    parms = load_parms(DATA_DIR + "parms.txt")

    for ind in range(parms['sequence_ind'][0], parms['sequence_ind'][1]):
        img_dir = DATA_DIR + 'img/' + str(ind) + '.png'
        txt_dir = DATA_DIR + 'txt/' + str(ind) + '.txt'

        log = load_txt(txt_dir, parms['past_time_interval'])
        hd_map = cv2.imread(img_dir)

        add_past_traj(hd_map, log, parms['past_time_interval'], parms['image_res'])

        cv2.imshow('input', hd_map)
        cv2.waitKey(100)
        a = 1



if __name__ == '__main__':
    main()