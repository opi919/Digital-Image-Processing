import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image1.png", cv2.IMREAD_GRAYSCALE)

copy_img = img.copy()

hist = np.zeros(256)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hist[img[i, j]] += 1

threshold_value = 100

_, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] >= threshold_value:
            copy_img[i, j] = 255
        else:
            copy_img[i, j] = 0

plt.subplot(4, 1, 1)
plt.imshow(img, cmap="gray")

plt.subplot(4, 1, 2)
plt.plot(hist)
plt.xlim([0, 256])

plt.subplot(4,1,3)
plt.imshow(binary_img,cmap='gray')

plt.subplot(4,1,4)
plt.imshow(copy_img,cmap='gray')

print(copy_img)
plt.show()

