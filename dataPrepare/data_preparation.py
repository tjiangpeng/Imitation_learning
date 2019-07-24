import sys
import os
import glob

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Imports -------------------------------------------------------------------
# ==============================================================================

import carla
import cv2
import logging
import argparse
import math

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================

DATA_DIR = "../../data/"
# DATA_DIR = "sample/"

# ==============================================================================
# -- Functuion -----------------------------------------------------------------
# ==============================================================================


def load_parms(dir, args):
    f = open(dir, "r")
    fl = f.readlines()

    args.pixels_per_meter = int(fl[1])
    args.pixels_ahead_hero = int(fl[3])
    args.past_time_interval = int(fl[5])
    args.image_res = [int(i) for i in fl[7].split(',')]
    args.sequence_ind = [int(i) for i in fl[9].split(',')]

    return args


def load_txt(dir, time):
    f = open(dir, "r")
    fl = f.readlines()

    data = dict()
    data['time'] = float(fl[1])
    data['hero_global_pos'] = [float(i) for i in fl[3].split(",")]
    data['road_id'] = int(fl[5])
    data['speed_limit'] = float(fl[7])
    data['speed'] = float(fl[9])
    data['traffic_light'] = fl[11][0:-1]
    data['translation_offset'] = [float(i) for i in fl[13].split(",")]
    data['angle'] = float(fl[15])

    data['hero_past_traj'] = []
    for pos in fl[17:(17+time)]:
        data['hero_past_traj'].append([int(i) for i in pos.split(",")])

    data['actor_past_traj'] = []
    num = int((len(fl) - 28) / (time+1))
    for ii in range(num):
        past_pos = []
        for pos in fl[(29+ii*(time+1)):(29+time+ii*(time+1))]:
            past_pos.append([int(i) for i in pos.split(",")])
        data['actor_past_traj'].append(past_pos)

    return data


def add_past_traj(hd_map, log, time, res):
    decrease = int(255 / (time + 1)) - 1
    color = [255-decrease, 255-decrease, 255]
    for past_pos in log['hero_past_traj']:
        if past_pos[0] >= 0 and past_pos[0] < res[0] and past_pos[1] >= 0 and past_pos[1] < res[1]:
            hd_map[past_pos[1]-1:past_pos[1]+2, past_pos[0]-1:past_pos[0]+2] = color
            # hd_map[past_pos[1], past_pos[0]] = [255, 255, 255]
        color[0] = color[0] - decrease
        color[1] = color[1] - decrease

    for vehicle_pos in log['actor_past_traj']:
        color = [255, 255 - decrease, 255 - decrease]
        for past_pos in vehicle_pos:
            if past_pos[0] >= 0 and past_pos[0] < res[0] and past_pos[1] >= 0 and past_pos[1] < res[1]:
                hd_map[past_pos[1] - 1:past_pos[1] + 2, past_pos[0] - 1:past_pos[0] + 2] = color
                # hd_map[past_pos[1], past_pos[0]] = [255, 255, 255]
            color[1] = color[1] - decrease
            color[2] = color[2] - decrease

    return hd_map


class World(object):
    def __init__(self, args, timeout):
        self.args = args
        self.timeout = timeout
        self.carla_world, self.carla_map = self._get_data_from_carla()

        waypoints = self.carla_map.generate_waypoints(2)
        margin = 50
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self._world_offset = (min_x, min_y)

        self.log = dict()
        for ind in range(self.args.sequence_ind[0], self.args.sequence_ind[1]):
            txt_dir = DATA_DIR + 'txt/' + str(ind) + '.txt'
            self.log[ind] = load_txt(txt_dir, args.past_time_interval)


    def _get_data_from_carla(self):
        try:
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(self.timeout)

            if self.args.map is None:
                world = self.client.get_world()
            else:
                world = self.client.load_world(self.args.map)

            town_map = world.get_map()
            return (world, town_map)

        except RuntimeError as ex:
            logging.error(ex)
            sys.exit()

    def transfer_to_img_coordinate(self, orig_pos, translation_offset, angle, img_res):
        pos = [0, 0]
        pos[0] = orig_pos[0] - translation_offset[0]
        pos[1] = orig_pos[1] - translation_offset[1]
        rotated_x = pos[0] * math.cos(angle / 180 * math.pi) - pos[1] * math.sin(angle / 180 * math.pi)
        rotated_y = pos[0] * math.sin(angle / 180 * math.pi) + pos[1] * math.cos(angle / 180 * math.pi)

        pos[0] = int(rotated_x + img_res[0] / 2 + 0.5)
        pos[1] = int(rotated_y + img_res[1] / 2 + 0.5)

        return pos

    def start(self):
        waypoints = self.carla_map.generate_waypoints(1)

        # map = cv2.imread('cache/no_rendering_mode/Town03.png')

        for wp in waypoints:
            pos = self.world_to_pixel(carla.Location(x=wp.transform.location.x, y=wp.transform.location.y))
            map[pos[1]-2:pos[1]+3, pos[0]-2:pos[0]+3] = [0, 255, 0]

        # map = cv2.resize(map, (1800, 1800))
        # cv2.imshow('map', map)
        # while True:
        #     k = cv2.waitKey(1)
        #     if k == 27:
        #         break

    def generate_routing(self, ind, img):
        waypoint = self.carla_map.get_waypoint(carla.Location(x=self.log[ind]['hero_global_pos'][0],
                                                              y=self.log[ind]['hero_global_pos'][1]))
        res = self.args.image_res
        for dist in range(1, 60):
            wp_next_list = waypoint.next(0.5 * dist)
            for wp_next in wp_next_list:
                pos = self.world_to_pixel(carla.Location(x=wp_next.transform.location.x,
                                                         y=wp_next.transform.location.y))
                pos = self.transfer_to_img_coordinate(pos, self.log[ind]['translation_offset'],
                                                      self.log[ind]['angle'], res)
                if pos[0] >= 0 and pos[0] < res[0] and pos[1] >= 0 and pos[1] < res[1]:
                    img[pos[1], pos[0]] = [0, 255, 0]

        return img



    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.args.pixels_per_meter * (location.x - self._world_offset[0])
        y = self.args.pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]


def data_process(args):
    world = World(args, timeout=2.0)

    for ind in range(args.sequence_ind[0], args.sequence_ind[1]):
        print(ind)

        # ind = 87
        img_dir = DATA_DIR + 'img/' + str(ind) + '.png'
        txt_dir = DATA_DIR + 'txt/' + str(ind) + '.txt'

        log = load_txt(txt_dir, args.past_time_interval)
        print("road id: ", log['road_id'])
        hd_map = cv2.imread(img_dir)

        add_past_traj(hd_map, log, args.past_time_interval, args.image_res)

        hd_map = world.generate_routing(ind, hd_map)

        cv2.imshow('input', hd_map)

        # print(log['traffic_light'])

        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break

        # cv2.waitKey(100)

    cv2.destroyAllWindows()


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

    args = argparser.parse_args()
    args.description = argparser.description

    args = load_parms(DATA_DIR + "parms.txt", args)

    data_process(args)


if __name__ == '__main__':
    main()