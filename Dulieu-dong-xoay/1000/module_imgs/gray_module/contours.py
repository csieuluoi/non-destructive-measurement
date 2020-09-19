import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from read_img import *
from image_processing import *

def get_bbox(roi_cord, bbox_size = (30, 20)):
    start_point, end_point = roi_cord
    min_x, min_y = start_point    
    max_x, max_y = end_point

    height = max_x - min_x
    width = max_y - min_y
    print('height:', height)
    print('width:', width)
    if bbox_size[0] > height:
        new_min_x = min_x - int((bbox_size[1] - height)/2)
    else:
        new_min_x = min_x
    if bbox_size[1] > width:
        new_min_y = min_y - int((bbox_size[0] - height)/2)
    else:
        new_min_y = min_y

    new_max_x = new_min_x + bbox_size[1]
    new_max_y = new_min_y + bbox_size[0]

    return ((new_min_x, new_min_y),(new_max_x, new_max_y))

def find_contour(img, c_threshold = (100, 200)):
    
    # if img == None:
    #     img = cv2.imread("input.png")
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # else: 
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    ret, gray = cv2.threshold(gray, c_threshold[0], c_threshold[1], 0)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = sorted(contours, key=cv2.contourArea)[-1]
    mask = np.zeros(img.shape, np.uint8)
    contour_img = cv2.drawContours(mask, [largest_area], 0, (255,255,255,255), -1)
    # contour_img = cv2.drawContours(mask, contours, -1, (255,255,255,255), -1)

    plt.imshow(contour_img)
    plt.show()
    dst = cv2.bitwise_and(img, mask)
    mask = 255 - mask
    roi = cv2.add(dst, mask)

    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.imshow('roi', roi)
    # cv2.waitKey(0)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(roi_gray, c_threshold[0], c_threshold[1], 0)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_x = 0
    max_y = 0
    min_x = img.shape[1]
    min_y = img.shape[0]

    for c in contours:
        if 100 > cv2.contourArea(c):
            x, y, w, h = cv2.boundingRect(c)
            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x+w, max_x)
            max_y = max(y+h, max_y)

    start_point = (min_x, min_y) 
    
    end_point = (max_x, max_y) 
    print('start_point:', start_point)
    print('end_point:', end_point)
    roi_cord = (start_point, end_point)
    start_point, end_point = get_bbox(roi_cord, bbox_size = (30, 20))

    print('start_point:', start_point)
    print('end_point:', end_point)

    
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 2
    
    # Using cv2.rectangle() method 
    # Draw a rectangle with blue line borders of thickness of 2 px 
    img = cv2.rectangle(img, start_point, end_point, color, thickness) 
    roi = roi[min_y:max_y, min_x:max_x]
    # cv2.imwrite("roi.png", roi)
    print(roi.shape)
    plt.imshow(img)
    plt.show()

def normalize_0_1(img):
    img = np.floor((img - np.min(img))/(np.max(img) - np.min(img)) * 255)
    return img.astype(np.uint8)



if __name__ == '__main__':
    col_keep_fraction = 0.1
    row_keep_fraction = 0.1
    # module_list, normalized_image_list = preprocess(denoised = True, col_keep_fraction = col_keep_fraction, row_keep_fraction = row_keep_fraction)
    files, img_list = load_module()    
    for module in img_list:
        gray = normalize_0_1(module)
        find_contour(gray, c_threshold= (0, 100))

    