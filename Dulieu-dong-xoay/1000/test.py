
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

img = cv2.imread('img.png', 0)
img_size = img.shape
x = np.arange(0, img_size[1])
y = np.arange(0, img_size[0])
X,Y = np.meshgrid(x,y)
Z =  img

print(X.shape)
print(Y.shape)
print(Z.shape)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')


# Plot a 3D surface
ax.plot_surface(X, Y, Z)


plt.show()

