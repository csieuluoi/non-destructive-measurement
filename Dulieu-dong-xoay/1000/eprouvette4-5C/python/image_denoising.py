import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pylab as pl

from scipy import fftpack
from PIL import Image, ImageFilter

from scipy import ndimage
from read_img import subplot

import os
import cv2

def denoising_im(im):
    im_fft = fftpack.fft2(im)
    # In the lines following, we'll make a copy of the original spectrum and
    # truncate coefficients.

    # Define the fraction of coefficients (in each direction) we keep
    # keep_fraction = 0.5
    col_keep_fraction = 0.5
    row_keep_fraction = 0.1
    # Call ff a copy of the original transform. Numpy arrays have a copy
    # method for this purpose.
    im_fft2 = im_fft.copy()
    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft2.shape

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[int(r*row_keep_fraction):int(r*(1-row_keep_fraction))] = 0

    # Similarly with the columns:
    im_fft2[:, int(c*col_keep_fraction):int(c*(1-col_keep_fraction))] = 0

    # Reconstruct the denoised image from the filtered spectrum, keep only the
    # real part for display.
    im_new = fftpack.ifft2(im_fft2).real

    return im_new


def plot_3d_compare(raw_im, denoised_im, title = ['raw', 'denoised']):
    raw_im_plot =  raw_im[::1,::1]
    denoised_im_plot = denoised_im[::1, ::1]

    fig = pl.figure(facecolor='w',figsize = (20, 10))
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    x,y = np.mgrid[:raw_im_plot.shape[0],:raw_im_plot.shape[1]]

    ax1.plot_surface(x,y,raw_im,cmap=pl.cm.jet,rstride=1,cstride=1,linewidth=0.,antialiased=False)
    ax1.set_title(title[0])

    ax2 = fig.add_subplot(1,2,2,projection='3d')
    x,y = np.mgrid[:denoised_im_plot.shape[0],:denoised_im_plot.shape[1]]
    ax2.plot_surface(x,y,denoised_im,cmap=pl.cm.jet,rstride=1,cstride=1,linewidth=0.,antialiased=False)
    ax2.set_title(title[1])
    # ax2.set_zlim3d(100,130)
    pl.show()

def plot_raw_2d_and_3d(raw_im, denoised_im):
    z = raw_im
    mydata = z[::1,::1]
    subplot(mydata, title = 'raw')

    z = denoised_im
    mydata = z[::1,::1]
    subplot(mydata, title = 'denoised')


def denoised_all_img(save_fold_name):

    cwd = os.getcwd()
    files = os.listdir(cwd)

    image_files = [file for file in files if file.endswith('part.tif')]
    real_images = []
    imagine_images = []
    for image_name in image_files:
        if image_name.endswith('realpart.tif'):
            # print(image_name[:-3])
            im = Image.open(image_name)
            denoised_im = denoising_im(np.asarray(im))
            # save all the denoised im to a folder
            save_dir = os.path.join(cwd, save_fold_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(os.path.join(save_dir, image_name[:-3] + 'png'), denoised_im)

            # plt.imsave(os.path.join(save_dir, image_name[:-3] + 'png'), denoised_im)
        else:
            im = Image.open(image_name)
            denoised_im = denoising_im(np.asarray(im))
            # save all the denoised im to a folder
            save_dir = os.path.join(cwd, save_fold_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(os.path.join(save_dir, image_name[:-3] + 'png'), denoised_im)

            # plt.imsave(os.path.join(save_dir, image_name[:-3] + 'png'), denoised_im)
    print('Done processing blurry images')


if __name__ == '__main__':
    im_name = '2A_500kHz_400u_realpart.tif'
    raw_im = plt.imread(im_name).astype(float)
    denoised_im = denoising_im(raw_im)

    # plot_raw_2d_and_3d(raw_im, denoised_im)

    # plot_3d_compare(raw_im, denoised_im)



    # for i in range(10):
    #     im_blur = ndimage.gaussian_filter(im, i)

    #     plot_raw_2d_and_3d(im, im_blur)

    denoised_all_img('denoised_images')



# % Calculation of image background:
# line_top = zeros(1,image_size(2)-4);
# line_bottom = line_top;

#     for col = 4:image_size(2)-1
#         for line = 11:25
#             line_top(:,col-3) = line_top(:,col-3)+ image_module(line,col);
#         end

#         for line = image_size(1)-24:image_size(1)-10
#             line_bottom(:,col-3) = line_bottom(:,col-3) + image_module(line,col);
#         end

#     end

#     line_top = line_top/15;
#     line_bottom = line_bottom/15;
#     line_average = 0.5*(line_top+line_bottom);

# % Lap ma tran anh nen:

#     for line = 1:image_size(1)-20
#         image_ground(line,:) = line_average;
#     end

# % Hieu chinh kich thuoc anh cu cho phu hop voi anh nen:
#     for line = 1:image_size(1)-20
#         for col = 4:image_size(2)-1
#             image_corr(line,col-3) = image_module(line+10,col);
#         end
#     end

# % Ma tran anh bo nen:
#     %     image_sub =  image_ground - image_corr;
#     image_sub =  abs(image_ground - image_corr);
#     figure()
#     mesh(image_sub);
#     title(['2A 500kHz ' num2str(2^var*100) 'µm -- Subtracted Module']);
