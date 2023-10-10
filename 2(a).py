import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg',0)

range_level = (0,150)
row,col = image.shape
output_image = image.copy()

for i in range(row):
    for j in range(col):
        if output_image[i][j]>range_level[0] and output_image[i][j]<range_level[1]:
            output_image[i][j]+=150

plt.subplot(2,1,1)
plt.imshow(image,cmap='gray')

plt.subplot(2,1,2)
plt.imshow(output_image,cmap='gray')
plt.show()

