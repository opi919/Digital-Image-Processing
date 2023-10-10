import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image2.jpg",0)

mask = 224

masked_image = image & mask

diff_image = image - masked_image

plt.subplot(3,1,1)
plt.imshow(image,cmap='gray')

plt.subplot(3,1,2)
plt.imshow(masked_image,cmap='gray')

plt.subplot(3,1,3)
plt.imshow(diff_image,cmap='gray')

plt.show()