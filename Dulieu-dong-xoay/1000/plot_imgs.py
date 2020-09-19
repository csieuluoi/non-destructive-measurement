import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

step = 15
img_size = (40,40)
X = np.load(f'train_data/{step}_{img_size}_train_cut_imgs.npy')
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


j = 0
for i in range(1, X.shape[0], 12):

    plot_grid(X[j:i])
    j = i
