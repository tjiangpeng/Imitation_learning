import os

DATA_DIR = "../../data/"

for filename in os.listdir(DATA_DIR + 'train/' + 'img'):
    num = int(filename[0:-4])
    newname = "{:0>7d}.png".format(num)

    src = DATA_DIR + 'train/' + 'img/' + filename
    dst = DATA_DIR + 'train/' + 'img/' + newname
    os.rename(src, dst)

for filename in os.listdir(DATA_DIR + 'train/' + 'txt'):
    num = int(filename[0:-4])
    newname = "{:0>7d}.txt".format(num)

    src = DATA_DIR + 'train/' + 'txt/' + filename
    dst = DATA_DIR + 'train/' + 'txt/' + newname
    os.rename(src, dst)