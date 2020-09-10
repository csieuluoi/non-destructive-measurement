import cv2
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



def save_img(img_list, type = 'module', length_list = ['200','400','800']):
	cwd = os.getcwd()
	fold_dir = os.path.join(cwd, type)
	if not os.path.exists(fold_dir):
		os.mkdir(fold_dir)
	for img, length in zip(img_list, length_list):
		cv2.imwrite(os.path.join(fold_dir, 'preprocessed_'+ type + '_' + length +'.png'), img)

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

def contour_cv2(module_img_list):
	kernel = np.ones((2,2), np.uint8)

	print([np.max(img) for img in module_img_list])
	for img in module_img_list:
		imgCanny = cv2.Canny(img,100, 200)
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

def rotate_img(image, angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]):
	imgs = []
	for a in angles:
		imgs.append(rotate_image(image, a))

	return imgs

def save_rotate_img(img_list, fold_name = 'rotated_img', length = '200' ):
	cwd = os.getcwd()
	print(cwd)
	fold_dir = os.path.join(cwd, fold_name)
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


def imshow(image):
	cv2.imshow('image', image)
	cv2.waitKey(0)
	# if cv2.waitKey(1) & 0xFF == ord('q'):

	# When everything done, release the capture
	# cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	module, normalized_image_list = preprocess(denoised = True, col_keep_fraction = col_keep_fraction, row_keep_fraction = row_keep_fraction)

	save_img(module)
	save_img(normalized_image_list, type = 'normalized_module')

	module_img_list = load_module('normalized_module')
	cut_imgs = [cut_image(img) for img in module_img_list]
	plt.figure(figsize = (12, 3))
	plt.subplot(141),plt.imshow(cut_imgs[0]),plt.title('200 um')
	plt.subplot(142),plt.imshow(cut_imgs[1]),plt.title('400 um')
	plt.subplot(143),plt.imshow(cut_imgs[2]),plt.title('800 um')
	plt.show()

	save_img(cut_imgs, type = 'train_img')


	cut_imgs = load_module('train_img')
	# resized_imgs = [cv2.resize(img, (20, 20)) for img in cut_imgs]
	lengths = ['150', '350', '700']
	for img, length in zip(cut_imgs, lengths):
		imgs = rotate_img(img)
		save_rotate_img(imgs, fold_name = 'rotated_imgs', length = length)
		print('save done')
		print(np.array(imgs).shape)
		plot_grid(imgs, nrows_ncols = (4, 3), figsize = (4., 4.))


