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
import numpy as np

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================

DATA_DIR = "../../data/"
# DATA_DIR = "sample/"

TIME_PAST = 10
TIME_FUTURE = 10

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'

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
    data['is_junction'] = fl[7][0:-1]
    data['speed_limit'] = float(fl[9])
    data['speed'] = float(fl[11])
    data['traffic_light'] = fl[13][0:-1]
    data['translation_offset'] = [float(i) for i in fl[15].split(",")]
    data['angle'] = float(fl[17])

    data['hero_past_traj'] = []
    for pos in fl[(19+time-TIME_PAST):(19+time)]:
        data['hero_past_traj'].append([int(i) for i in pos.split(",")])

    data['actor_past_traj'] = []
    num = int((len(fl) - 20 - time) / (time+1))
    for ii in range(num):
        past_pos = []
        for pos in fl[(21+time+time-TIME_PAST+ii*(time+1)):(21+time+time+ii*(time+1))]:
            past_pos.append([int(i) for i in pos.split(",")])
        data['actor_past_traj'].append(past_pos)

    return data


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
        for ind in range(self.args.sequence_ind[0], self.args.sequence_ind[1]+1):
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

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.args.pixels_per_meter * (location.x - self._world_offset[0])
        y = self.args.pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def transfer_to_img_coordinate(self, orig_pos, translation_offset, angle, img_res):
        pos = [0, 0]
        pos[0] = orig_pos[0] - translation_offset[0]
        pos[1] = orig_pos[1] - translation_offset[1]
        rotated_x = pos[0] * math.cos(angle / 180 * math.pi) - pos[1] * math.sin(angle / 180 * math.pi)
        rotated_y = pos[0] * math.sin(angle / 180 * math.pi) + pos[1] * math.cos(angle / 180 * math.pi)

        pos[0] = int(rotated_x + img_res[0] / 2 + 0.5)
        pos[1] = int(rotated_y + img_res[1] / 2 + 0.5)

        return pos

    def global_to_img_coordinate(self, pos, ind):
        pos = self.world_to_pixel(carla.Location(x=pos[0], y=pos[1]))
        pos = self.transfer_to_img_coordinate(pos, self.log[ind]['translation_offset'],
                                              self.log[ind]['angle'], self.args.image_res)

        return pos

    def start(self):
        waypoints = self.carla_map.generate_waypoints(1)

        # map = cv2.imread('cache/no_rendering_mode/Town03.png')

        for wp in waypoints:
            pos = self.world_to_pixel(carla.Location(x=wp.transform.location.x, y=wp.transform.location.y))
            map[pos[1]-2:pos[1]+3, pos[0]-2:pos[0]+3] = [0, 255, 0]


    def generate_routing(self, ind, img):
        res = self.args.image_res
        # Find the future pos
        is_outside = False
        i_f = ind
        continue_frame = 0
        while continue_frame < 2:
            i_f = i_f + 1
            if i_f > self.args.sequence_ind[1]:
                print("Reach the final frame!")
                return False, img

            pos_f = self.log[i_f]['hero_global_pos']
            pos = self.global_to_img_coordinate(pos_f, ind)
            if pos[0] < 0 or pos[0] >= res[0] or pos[1] < 0 or pos[1] >= res[1]:
                is_outside = True

            is_junction = self.log[i_f]['is_junction']

            if is_outside and is_junction == 'False':
                continue_frame = continue_frame + 1

        road_id_f = self.log[i_f]['road_id']

        # Generate routing
        # a. find the waypoint containing the future point
        for i in range(ind, ind-50, -1):
            waypoint = self.carla_map.get_waypoint(carla.Location(x=self.log[i]['hero_global_pos'][0],
                                                                  y=self.log[i]['hero_global_pos'][1]))
            for dist_f in range(22, 120):
                wp_next_list = waypoint.next(dist_f)
                flag = False
                for wp_next in wp_next_list:
                    dist = math.sqrt((self.log[i_f]['hero_global_pos'][0] - wp_next.transform.location.x)**2 +
                                     (self.log[i_f]['hero_global_pos'][1] - wp_next.transform.location.y)**2)
                    if wp_next.road_id == road_id_f and dist <= 2:
                        # must be outsides the view
                        pos = [wp_next.transform.location.x, wp_next.transform.location.y]
                        pos = self.global_to_img_coordinate(pos, ind)
                        if pos[0] < 0 or pos[0] >= res[0] or pos[1] < 0 or pos[1] >= res[1]:
                            flag = True
                if flag:
                    break
            if flag:
                break

        # b. check the traffic light and speed limit
        if self.log[ind]['traffic_light'] == 'Red':
            color = [255, 0, 255]
        elif self.log[ind]['traffic_light'] == 'Yellow':
            color = [0, 255, 255]
        else:
            if self.log[ind]['speed_limit'] == 30:
                color = [0, 255, 0]
            elif self.log[ind]['speed_limit'] == 60:
                color = [80, 255, 80]
            else:
                color = [160, 255, 160]

        # c. extract the waypoint as routing
        previous_flag = True
        previous_point = (352, 460)
        for dist in range(6, 200):
            dist = dist * 0.5
            wp_next_list = waypoint.next(dist)

            if len(wp_next_list) > 1:
                if dist >= dist_f:
                    break
                for wp_next in wp_next_list:
                    wp_next_next_list = wp_next.next(dist_f - dist)
                    flag = False
                    for wp_next_next in wp_next_next_list:
                        if wp_next_next.road_id == road_id_f:
                            flag = True
                            break
                    if flag:
                        break
            else:
                wp_next = wp_next_list[0]

            pos = [wp_next.transform.location.x, wp_next.transform.location.y]
            pos = self.global_to_img_coordinate(pos, ind)

            # Remove some previous points
            if pos[1] <= 460 and previous_flag:
                previous_flag = False
            if pos[1] > 460 and previous_flag:
                continue

            if pos[0] >= 0-8 and pos[0] < res[0]+8 and pos[1] >= 0-8 and pos[1] < res[1]+8:
                # img[pos[1]-2:pos[1]+3, pos[0]-2:pos[0]+3] = color
                cv2.line(img, previous_point, (pos[0], pos[1]), color, 5)
                previous_point = (pos[0], pos[1])

        return True, img

    def add_past_traj(self, hd_map, ind):
        res = self.args.image_res

        decrease = int(255 / (TIME_PAST + 1)) - 1
        color = [255 - decrease, 255 - decrease, 255]
        for past_pos in self.log[ind]['hero_past_traj']:
            if past_pos[0] >= 0 and past_pos[0] < res[0] and past_pos[1] >= 0 and past_pos[1] < res[1]:
                hd_map[past_pos[1] - 2:past_pos[1] + 3, past_pos[0] - 2:past_pos[0] + 3] = color
                # hd_map[past_pos[1], past_pos[0]] = [255, 255, 255]
            color[0] = color[0] - decrease
            color[1] = color[1] - decrease

        for vehicle_pos in self.log[ind]['actor_past_traj']:
            color = [255, 255 - decrease, 255 - decrease]
            for past_pos in vehicle_pos:
                if past_pos[0] >= 0 and past_pos[0] < res[0] and past_pos[1] >= 0 and past_pos[1] < res[1]:
                    hd_map[past_pos[1] - 2:past_pos[1] + 3, past_pos[0] - 2:past_pos[0] + 3] = color
                    # hd_map[past_pos[1], past_pos[0]] = [255, 255, 255]
                color[1] = color[1] - decrease
                color[2] = color[2] - decrease

        return hd_map

    def get_future_traj(self, ind):
        res = self.args.image_res
        traj = []
        img = np.zeros((self.args.image_res[0], self.args.image_res[1], 3), np.uint8)
        for i in range(ind+1, ind+TIME_FUTURE+1):
            pos = self.log[i]['hero_global_pos']
            pos = self.global_to_img_coordinate(pos, ind)
            if pos[0] >= 0 and pos[0] < res[0] and pos[1] >= 0 and pos[1] < res[1]:
                traj.append(pos)
                img[pos[1] - 1:pos[1] + 2, pos[0] - 1:pos[0] + 2] = [255, 255, 255]
        cv2.imshow('future trajectory', img)

        return traj


def data_process(args):
    world = World(args, timeout=2.0)
    outVideo = cv2.VideoWriter('../../data/out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                               (args.image_res[0], args.image_res[1]))

    for ind in range(args.sequence_ind[0], args.sequence_ind[1]):
    # for ind in range(684, args.sequence_ind[1]):
        print(ind)
        # ind = 4035

        img_dir = DATA_DIR + 'img/' + str(ind) + '.png'
        inputImg = cv2.imread(img_dir)

        traj = world.get_future_traj(ind)
        inputImg = world.add_past_traj(inputImg, ind)
        stopToken, inputImg = world.generate_routing(ind, inputImg)

        if not stopToken:
            break
        cv2.imshow("input", inputImg)
        outVideo.write(inputImg)

        # while True:
        #     k = cv2.waitKey(1)
        #     if k == 27:
        #         break

        cv2.waitKey(1)

    outVideo.release()
    cv2.destroyAllWindows()


def convert_to_tf_records(raw_data_dir):
    """ Convert the raw dataset into TF-Record dumps"""
    training_files = tf.gfile.Glob(
        os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', )
    )

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

    # data_process(args)

    # Convert the raw data into tf-records
    training_records, validation_records = convert_to_tf_records(DATA_DIR)


if __name__ == '__main__':
    main()