import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from read_img import *
from mpl_toolkits.axes_grid1 import ImageGrid

import os
col_keep_fraction = 0.1
row_keep_fraction = 0.1

# cwd = os.getcwd()
# files = os.listdir(cwd)

# image_files = [file for file in files if file.endswith('part.tif')]
# real_images = []
# imagine_images = []
# for image_name in image_files:
# 	img = cv2.imread(image_name, 0)
# 	if image_name.endswith('realpart.tif'):
# 		# print(image_name[:-3])
# 		print(img.shape)
# 		real_images.append(img)
# 		# denoised_im = denoising_im(np.asarray(im))
# 		# save all the denoised im to a folder
# 		# save_dir = os.path.join(cwd, save_fold_name)
# 	else:
# 		imagine_images.append(img)



def save_img(img_list, type = 'module', length_list = ['200','400','800'], color_convert = False):
	if color_convert:
		# print([img for img in img_list])

		img_list = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB) for img in img_list]
	cwd = os.getcwd()
	fold_dir = os.path.join(cwd, 'generated_imgs', type)
	if not os.path.exists(fold_dir):
		os.mkdir(fold_dir)
	for img, length in zip(img_list, length_list):
		cv2.imwrite(os.path.join(fold_dir, length + '_' + type +'.png'), img)

def load_module(folder_name = 'module'):
	module_img_list = []
	cwd = os.getcwd()
	module_dir = os.path.join(cwd, folder_name)
	files = os.listdir(module_dir)

	image_files = [os.path.join(module_dir, file) for file in files if file.endswith('.png')]

	for image_name in image_files:
		img = cv2.imread(image_name, 0)
		module_img_list.append(img)
		# print(img.shape)
	return module_img_list

def cut_image(img):
	if img.shape[0] > img.shape[1]:
		lines = (18 , 42)
		cols = (0, 20)
	else:
		lines=  (16, 40)
		cols = (18, 38)

	img = img[lines[0]:lines[1], cols[0]: cols[1]]

	return img


def padding(img, full_size = (85, 85), padding_type = 'edge'):
	ix,iy = img.shape
	num_pad_rows = int((full_size[0] - ix)/2)
	num_pad_cols = int((full_size[1] - iy)/2)
	img = np.pad(img, ((num_pad_rows, num_pad_rows), (num_pad_cols, num_pad_cols)), mode = padding_type)

	return img



def cut_out_hole(img, img_size = (30, 20), type = 'normalized'):

	mid_point = ((img.shape[0]/2), int(img.shape[1]/2))
	if type == 'normalized':
		row_arr, col_arr = np.where(img==np.amax(img))
		middle_point_ind = (int(row_arr.shape[0]/2), int(col_arr.shape[0]/2))
		row_ind, col_ind = row_arr[middle_point_ind[0]], col_arr[middle_point_ind[1]]
	else:
		row_arr, col_arr = np.where(img==np.amin(img))
		middle_point_ind = (int(row_arr.shape[0]/2), int(col_arr.shape[0]/2))
		row_ind, col_ind = row_arr[middle_point_ind[0]], col_arr[middle_point_ind[1]]
	half_size_row = int(img_size[0]/2)
	half_size_col = int(img_size[1]/2)

	if img.shape == (61, 21)  and img_size[1] > 14:
		img = img[row_ind - half_size_row: row_ind + half_size_row, int((img.shape[1] - img_size[1])/2): img.shape[1] -int((col_ind - img_size[1])/2)]
	else:
		img = img[row_ind - half_size_row: row_ind + half_size_row, col_ind - half_size_col: col_ind + half_size_col]

	return img


def contour_cv2(module_img_list):
	kernel = np.ones((2,2), np.uint8)

	print([np.max(img) for img in module_img_list])
	for img in module_img_list:
		imgCanny = cv2.Canny(img, 100, 200)
		ret, thresh = cv2.threshold(img, 10, 40, cv2.THRESH_BINARY)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
		thresh = cv2.dilate(thresh, kernel, iterations = 5)
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# Vì drawContours sẽ thay đổi ảnh gốc nên cần lưu ảnh sang một biến mới.
		imgOrigin = img.copy()
		img1 = img.copy()
		img2 = img.copy()

		# Vẽ toàn bộ contours trên hình ảnh gốc
		cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)

		# Vẽ chỉ contour thứ 4 trên hình ảnh gốc
		cv2.drawContours(img2, contours, 3, (0, 255, 0), 3)

		plt.figure(figsize = (12, 3))
		plt.subplot(141),plt.imshow(imgOrigin),plt.title('Original')
		# plt.xticks([]), plt.yticks([])
		plt.subplot(142),plt.imshow(imgCanny),plt.title('Canny Binary Image')
		# plt.xticks([]), plt.yticks([])
		plt.subplot(143),plt.imshow(img1),plt.title('All Contours')
		# plt.xticks([]), plt.yticks([])
		plt.subplot(144),plt.imshow(img2),plt.title('Contour 4')
		# plt.xticks([]), plt.yticks([])
		plt.show()


"""==========================================================================================="""
"""Data augmentation functions"""

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotate_img(image, angle_step = 30):
	imgs = []
	for angle in range( 0, 360, angle_step):
		## using rotate_image function from https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
		# imgs.append(rotate_image(image, angle))
		## using rotate_bound method from imutils package
		imgs.append(imutils.rotate_bound(image, angle))

	return imgs

def save_rotate_img(img_list, fold_name = 'rotated_img', length = '200' ):
	cwd = os.getcwd()
	if not os.path.exists(os.path.join(cwd, 'generated_imgs')):
		os.mkdir(os.path.join(cwd, 'generated_imgs'))

	fold_dir = os.path.join(cwd, 'generated_imgs', fold_name)
	if not os.path.exists(fold_dir):
		os.mkdir(fold_dir)
	for i, img in enumerate(img_list):
		cv2.imwrite(os.path.join(fold_dir, length +f'um_{i}.png'), img)


"""==========================================================================================="""
"""visualization functions"""
def plot_grid(imgs, nrows_ncols = (4, 3), figsize = (4, 3)):
	fig = plt.figure(figsize=figsize)
	grid = ImageGrid(fig, 111,  # similar to subplot(111)
					 nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
					 axes_pad=0.1,  # pad between axes in inch.
					 )

	for ax, im in zip(grid, imgs):
		# Iterating over the grid returns the Axes.
		ax.imshow(im)

	plt.show()


def preparing_images(imgs, step = 15, lengths = ['200', '400', '750'], cut_img_size = None, padding_size = (90, 90), padding_type = 'edge', output_img_size = (20, 20), save = False, plot = False):
	if cut_img_size != None:
		imgs = [cut_out_hole(img, img_size = cut_img_size) for img in imgs]
	# # padding before rotate
	padded_imgs = [padding(img, full_size = padding_size, padding_type = padding_type) for img in imgs]

	# cut_imgs = [cut_out_hole(img, img_size = (20, 20)) for img in normalized_image_list]

	for i, (img, length) in enumerate(zip(padded_imgs, lengths)):
		imgs = rotate_img(img, angle_step = step)
		imgs = [cut_out_hole(img, img_size = output_img_size) for img in imgs]
		imgs = [255 - img for img in imgs]
		if save:
			save_rotate_img(imgs, fold_name = f'{step}_{output_img_size}_rotated_cut_imgs', length = length)
			print(f'save {i}-step-{step} done')
		# print(np.array(imgs).shape)
		if plot:
			plot_grid(imgs, nrows_ncols = (6, 4), figsize = (4., 4.))



if __name__ == '__main__':

	# module_list, normalized_image_list = preprocess(denoised = True, col_keep_fraction = col_keep_fraction, row_keep_fraction = row_keep_fraction)


	# # save_img(module_list)
	# # save_img(normalized_image_list, type = 'normalized_module')

	# # module_img_list = load_module('normalized_module')
	# # module_img_list = load_module('module')

	# """========================================================================================================"""
	# """ test cut cut_out_hole function: """
	# """========================================================================================================"""
	# # for img in module_img_list:
	# # 	cut_img = cut_out_hole(img, img_size = (30, 20), type = 'module')
	# # 	plt.imshow(cut_img)
	# # 	plt.show()
	# """========================================================================================================"""
	# """plot cut images"""
	# """========================================================================================================"""

	# # plt.figure(figsize = (12, 3))
	# # plt.subplot(141),plt.imshow(normalized_image_list[0]),plt.title('200 um')
	# # plt.subplot(142),plt.imshow(normalized_image_list[1]),plt.title('400 um')
	# # plt.subplot(143),plt.imshow(normalized_image_list[2]),plt.title('800 um')
	# # plt.show()

	# # plt.figure(figsize = (12, 3))
	# # plt.subplot(141),plt.imshow(module_list[0]),plt.title('200 um')
	# # plt.subplot(142),plt.imshow(module_list[1]),plt.title('400 um')
	# # plt.subplot(143),plt.imshow(module_list[2]),plt.title('800 um')
	# # plt.show()

	# # # cut_imgs = [cut_out_hole(img, img_size = (20, 20)) for img in normalized_image_list]
	# output_img_size = (40, 40)
	# steps = [10, 15, 25, 29]
	lengths = ['150', '350', '700']


	# print([img.shape for img in normalized_image_list])
	# padding_types = ['edge', 'linear_ramp']
	# # for padding_type in padding_types:
	# for step in steps:
	# 	# preparing_images(normalized_image_list, step = step, lengths = lengths, cut_img_size = (20, 16),
	# 	# 	padding_size = (90, 90), padding_type = padding_types[0], output_img_size = (20, 20), save = True)

	# 	preparing_images(normalized_image_list, step = step, lengths = lengths, cut_img_size = None,
	# 		padding_size = (90, 90), padding_type = padding_types[0], output_img_size = output_img_size, save = True, plot = False)
	module_list, normalized_image_list = preprocess(denoised = True, col_keep_fraction = col_keep_fraction, row_keep_fraction = row_keep_fraction)
	save_img(module_list, type = 'gray_module', length_list = lengths, color_convert = False)
	save_img(normalized_image_list, type = 'gray_normalized_module', length_list = lengths, color_convert = True)

	module_list, normalized_image_list = preprocess(denoised = False, col_keep_fraction = col_keep_fraction, row_keep_fraction = row_keep_fraction)
	save_img(module_list, type = 'raw_gray_module', length_list = lengths, color_convert = False)
	save_img(normalized_image_list, type = 'raw_gray_normalized_module', length_list = lengths, color_convert = True)
