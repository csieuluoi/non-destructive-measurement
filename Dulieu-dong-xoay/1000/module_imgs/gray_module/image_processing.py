from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as pl
from matplotlib import pyplot as plt
from PIL import Image
from scipy import fftpack

import numpy as np
import os

from read_img import *


def load_module():
    cwd = os.getcwd()
    files = os.listdir(cwd)
    img_list = []
    for file in files:
        if file.endswith('.png'):
            img = np.asarray(Image.open(os.path.join(cwd, file)))
            img_list.append(img)
    return  (files, img_list)


if __name__ == '__main__':
    files, img_list = load_module()
    print(files)

    plot_raw_2d_and_3d(img_list, imagine_images = None, title = ['processed modules', ''])
