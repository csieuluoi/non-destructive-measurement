import os
import numpy as np
import cv2


cwd = os.getcwd()
fold_names = os.listdir(cwd)
specimens = []
lengths = []
for fold in fold_names:
    # print(fold)
    if fold[:9] == 'eprouvett':
        fold_dir = os.path.join(cwd, fold, 'python', 'rotated_imgs')
        images_name = os.listdir(fold_dir)
        # print(f'\nfile in folder {fold}: ')
        # print(images_name)
        img_list = []
        length_list = []
        for name in images_name:
            img = cv2.imread(os.path.join(fold_dir, name), 0)
            img_list.append(img)
            length_list.append(int(name[:3]))
        specimens += img_list
        lengths += length_list
        print(specimens)
        print(lengths)

np.save('train_imgs.npy', specimens)
np.save('labels.npy', lengths)
