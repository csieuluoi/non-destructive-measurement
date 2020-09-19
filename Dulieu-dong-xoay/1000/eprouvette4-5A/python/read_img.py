
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as pl
from matplotlib import pyplot as plt
from PIL import Image
from scipy import fftpack

import numpy as np
import os


"""====================================================================================================="""
"""====================================================================================================="""
"""helper function to plot 2d and 3d graph"""
def subplot(mydata, title = 'real'):
	fig = pl.figure(facecolor='w',figsize = (20, 10))
	ax1 = fig.add_subplot(1,2,1)
	im = ax1.imshow(mydata,interpolation='nearest',cmap='binary')
	ax1.set_title('2D')

	ax2 = fig.add_subplot(1,2,2,projection='3d')
	x,y = np.mgrid[:mydata.shape[0],:mydata.shape[1]]
	ax2.plot_surface(x,y,mydata,cmap=pl.cm.jet,rstride=1,cstride=1,linewidth=0.,antialiased=False)
	ax2.set_title('3D ' + title + ' part')
	# ax2.set_zlim3d(100,130)
	pl.show()

def plot_raw_2d_and_3d(real_images, imagine_images = None, title = ['real', 'imagine']):

	for imageArr in real_images:
		z = imageArr
		mydata = z[::1,::1]
		subplot(mydata, title = title[0])

	if imagine_images is not None:
		for imageArr in imagine_images:
			z = imageArr
			mydata = z[::1,::1]
			subplot(mydata, title = title[1])


# def plot_remove_noises_2d_and_3d(real_images, imagine_images = None, img_size = (60,60), title = ['real', 'imagine']):
# 	for imageArr in real_images:
# 		real_ground_pix = calculate_ground_pix(imageArr)

# 		# remove surface noises
# 		imageArr = center_cut(imageArr, img_size = img_size)
# 		# imageArr = conservative_smoothing_gray(imageArr,5)
# 		# imageArr = np.asarray(Image.fromarray(imageArr).filter(ImageFilter.BLUR))
# 		# imageArr = normalize_img(imageArr, real_ground_pix)
# 		z = imageArr
# 		mydata = z[::1,::1]
# 		subplot(mydata, title =  title[0])

# 	if imagine_images is not None:
# 		for imageArr in imagine_images:
# 			imagine_ground_pix = calculate_ground_pix(imageArr)

# 			imageArr = center_cut(imageArr, img_size = img_size)
# 			# imageArr = conservative_smoothing_gray(imageArr,5)
# 			# imageArr = np.asarray(Image.fromarray(imageArr).filter(ImageFilter.BLUR))

# 			# imageArr = normalize_img(imageArr, imagine_ground_pix)
# 			z = imageArr
# 			mydata = z[::1,::1]
# 			subplot(mydata, title = title[1])


"""====================================================================================================="""
"""====================================================================================================="""
"""function to remove noise from raw images"""

def denoising_im(im, col_keep_fraction = 0.5, row_keep_fraction = 0.1):
    im_fft = fftpack.fft2(im)
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
    denoised_im = fftpack.ifft2(im_fft2).real

    return denoised_im

"""====================================================================================================="""
"""====================================================================================================="""
"""Function for data preprocessing"""

def read_data(fold_name = '', file_extension = 'part.tif'):
	"""
	Read data images from folder with extension
	Variables:
	fold_name (str): name of the data folder
	file_extension (str): file extension to read ('part.tif' or 'part.png')
	return: 2 lists contain real images and imagine images
	"""
	cwd = os.getcwd()
	files = os.listdir(os.path.join(cwd, fold_name))
	image_files = [file for file in files if file.endswith('part.tif')]
	real_images = []
	imagine_images = []
	for image in image_files:
		if image.endswith('realpart.tif'):
			real_images.append(np.asarray(Image.open(os.path.join(cwd, fold_name, image))))
		else:
			imagine_images.append(np.asarray(Image.open(os.path.join(cwd, fold_name, image))))

	return real_images, imagine_images

def cal_complex(real_images, imagine_images, denoised = False, col_keep_fraction = 0.5, row_keep_fraction = 0.1):
	"""
	Compute modules and arguments of complex images
	Variables:
	real_images (list): list of the real images
	imagine_images (list): list of the imagine images
	denoised (bool): True if denoise and vise versa, default = False
	col_keep_fraction (float): keep ratio for columns - in range (0.1, 1))
	row_keep_fraction (float): keep ratio for rows - in range (0.1, 1))
	return: 2 lists contain modules and arguments
	"""
	complex_images = [real_images[i] + imagine_images[i]*1j for i in range(len(real_images))]


	if denoised:
		module = [denoising_im(np.abs(img), col_keep_fraction, row_keep_fraction) for img in complex_images]
		argument = [denoising_im(np.angle(img), col_keep_fraction, row_keep_fraction) for img in complex_images]
	else:
		module = [np.abs(img) for img in complex_images]
		argument = [np.angle(img) for img in complex_images]
	return module, argument


def center_cut(imageArr, n_rows: tuple, n_cols: tuple):
	"""
	Cut out some rows and cols from an image
	Variables:
	imageArr (numpy array): image to cut
	n_rows (int): # of rows to cut
	n_cols (int): # of cols to cut
	return: image after cut out some rows and cols
	"""
	img_rows, img_cols = imageArr.shape
	# new_img_rows, new_img_cols = img_size
	# if img_rows > new_img_rows:
	# 	n_rows = int((img_rows - new_img_rows)/2)
	# else:
	# 	n_rows = 5
	# if img_cols > new_img_cols:
	# 	n_cols = int((img_cols - new_img_cols)/2)
	# else:
	# 	n_cols = 5
	imageArr = imageArr[n_rows[0]: img_rows - n_rows[1], n_cols[0]: img_cols - n_cols[1]]

	return imageArr

def cal_image_background(imageArr, n_lines_top, n_lines_bottom):
	"""
	Calculate the background matrix
	Variables:
	imageArr (numpy array): raw image
	n_lines_top (int): # of rows at top of the image used to calculate
	n_cn_lines_bottomols (int): # of rows at the bottom of the image used to calculate
	return: background matrix
	"""
	img_rows, img_cols = imageArr.shape

	line_top = np.zeros((img_cols))
	line_bottom = np.zeros((img_cols))

	for i in range(n_lines_top):
		line_top += imageArr[i, :]
	for i in range(n_lines_bottom):
		line_bottom += imageArr[img_rows -1 - i, :]

	line_top = line_top/n_lines_top
	line_bottom = line_bottom/n_lines_bottom
	line_average = 0.5*(line_top+line_bottom)

	# line_average = (line_top+line_bottom) / (n_lines_bottom + n_lines_top)
	ground_matrix = np.tile(line_average, (imageArr.shape[0], 1))

	return ground_matrix



def normalize_image(imageArr, ground_matrix):
	"""
	normalize the raw image by removing background...
	Variables:
	imageArr (numpy array): raw image
	ground_matrix (numpy array): background matrix
	return: normalized image
	"""
	# print("ground matrix shape: ", ground_matrix.shape)
	# print("imageArr matrix shape: ", imageArr.shape)
	# print(ground_matrix)
	normalized_image = np.abs(ground_matrix - imageArr)
	# normalized_image = np.abs(imageArr - ground_matrix)

	return normalized_image

def preprocess(denoised = False, col_keep_fraction = 0.5, row_keep_fraction = 0.1):
	real_images, imagine_images = read_data()
	modules, argument = cal_complex(real_images, imagine_images, denoised = denoised, col_keep_fraction = col_keep_fraction, row_keep_fraction=row_keep_fraction)
	normalized_image_list = []
	modules_cut_list = []
	# set config for each image (200um, 400um and 800um) following 'Draw_2A_500kHz.m'
	center_cut_config_list = [[(10, 10), (3, 1)], [(3, 7), (19, 1)], [(3, 3), (19, 1)]]
	cal_image_background_config_list = [(15, 15), (12, 12), (12, 12)]
	for i, imageArr in enumerate(modules):
		print(imageArr.shape)
		imageArr = center_cut(imageArr, n_rows = center_cut_config_list[i][0], n_cols = center_cut_config_list[i][1])
		modules_cut_list.append(imageArr)
		ground_matrix = cal_image_background(imageArr, n_lines_top = cal_image_background_config_list[i][0], n_lines_bottom = cal_image_background_config_list[i][1])
		normalized_image = normalize_image(imageArr, ground_matrix)
		normalized_image_list.append(normalized_image)

	return modules_cut_list, normalized_image_list

"""
def calculate_ground_pix(images):
	# Calculate the ground pixel
	img_vec = []
	for img in images:
		img_vec += list(np.squeeze(img.reshape(-1, 1)))
	uniqueArr, uniqeCount = np.unique(img_vec, return_index=False, return_inverse=False, return_counts=True)
	ground_pix = uniqueArr[np.argmax(uniqeCount)]

	return ground_pix

def normalize_img(imageArr, ground_pix = 128, reversed = True):
	imageArr = np.where(imageArr < ground_pix, ground_pix, imageArr)

	if reversed:
		imageArr = 255 - imageArr
	return imageArr

def normalize_imagine_img(imageArr):
	imageArr = np.abs(122.5 - imageArr)
	return imageArr
"""



if __name__ == '__main__':
	modules, normalized_image_list = preprocess(denoised = True, col_keep_fraction = 0.1, row_keep_fraction = 0.1)
	# # show raw modules and processed modules
	# plot_raw_2d_and_3d(modules, imagine_images = normalized_image_list, title = ['raw modules', 'processed modules'])

	# show only processed modules
	plot_raw_2d_and_3d(normalized_image_list, imagine_images = None, title = ['processed modules', ''])




















