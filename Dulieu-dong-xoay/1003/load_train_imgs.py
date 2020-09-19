import os
import numpy as np
import cv2

steps = [10, 15, 25, 29]

cwd = os.getcwd()
fold_names = os.listdir(cwd)

for step in steps:
    specimens = []
    lengths = []
    for fold in fold_names[:8]:
        # print(fold)
        # if fold[:9] == 'eprouvett':
        fold_dir = os.path.join(cwd, fold, 'python/generated_imgs', f'{step}_rotated_cut_imgs')
        images_name = os.listdir(fold_dir)
        # print(f'\nfile in folder {fold}: ')
        # print(images_name)
        img_list = []
        length_list = []
        for name in images_name:
            img = cv2.imread(os.path.join(fold_dir, name), 0)
            img_list.append(img)
            try:
                label = float(name[:4])
            except:
                label = float(name[:3])
            length_list.append(label)
        specimens += img_list
        lengths += length_list
        # print(specimens)
        # print(length_list)

    print(f'{step}- {len(lengths)}')
    np.save(f'train_data/{step}_train_cut_imgs.npy', specimens)
    np.save(f'train_data/{step}_labels_cut.npy', lengths)
