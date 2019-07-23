import cv2

DATA_DIR = "../../data/"


def load_txt(dir):
    f = open(dir, "r")
    fl = f.readlines()

    data = {}
    data['time'] = float(fl[1])
    data['hero_global_pos'] = [float(i) for i in fl[3].split(",")]
    data['road_id'] = int(fl[5])
    data['speed_limit'] = int(fl[7])
    data['speed'] = float(fl[9])
    data['translation_offset'] = [float(i) for i in fl[11].split(",")]
    data['angle'] = float(fl[13])

    a = 1



def main():
    load_txt(DATA_DIR + "txt/0.txt")


if __name__ == '__main__':
    main()