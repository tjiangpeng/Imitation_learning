import numpy as np

from hparms import *

IMAGE_TO_REAL_SCALE = 0.2


def agent_cord_to_image_cord(pos: np.ndarray):
    pos = pos / IMAGE_TO_REAL_SCALE
    pos[1] = pos[1] * -1
    pos = pos + [IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2] + 0.5

    return pos.astype(np.int)