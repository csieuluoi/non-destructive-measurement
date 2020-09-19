from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as pl
from matplotlib import pyplot as plt
from PIL import Image
from scipy import fftpack

import numpy as np
import os
import cv2

from read_img import *

def gausian_filter(img):
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)
    blur = cv2.blur(img, (5, 5))
    gblur = cv2.GaussianBlur(img, (5, 5), 0)

    titles = ['image', '2D Convolution', 'blur', 'GaussianBlur']
    images = [img, dst, blur, gblur]
    fig, axs = plt.subplots(1,4, figsize=(12, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axs.ravel()
    for i in range(4):
        axs[i].imshow(images[i], cmap = 'jet')
        axs[i].set_title(titles[i])
    plt.show()

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

    # plot_raw_2d_and_3d(img_list, imagine_images = None, title = ['processed modules', ''])
    gausian_filter(img_list[0])
