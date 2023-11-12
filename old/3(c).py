import matplotlib.pyplot as plt
import numpy as np
import cv2
import random


def harmonic_filter(image, x, y, a, b, totalPixels):
   result = 0
   for i in range(-a, a):
       for j in range (-b, b):
           if image[x+i, y+j]>0:
               result += 1/image[x+i, y+j]


   return totalPixels/result


def geometric_mean(image, x, y, a, b, totalPixels):
   result = np.float64(1)
   validPixel = 0
   for i in range(-a, a):
       for j in range(-b, b):
           if image[x+i, y+j]>0:
               validPixel += 1
               result *= image[x+i, y+j]
   # print(result ** (1/totalPixels))
   if (validPixel>0):
       return result ** (1/totalPixels)
   else:
       return result


original_image = cv2.imread("image1.png", cv2.IMREAD_GRAYSCALE)
dup_img = original_image.copy()
noise = 0.02
row, col = dup_img.shape
for i in range(row):
   for j in range(col):
       rand = random.random()
       if rand < noise/2:
           dup_img[i, j] = 0
       elif rand < noise:
           dup_img[i, j] = 255


harmonic_mean_filter_image = dup_img.copy()
geometric_mean_filter_image = dup_img.copy()
row, col = dup_img.shape


for i in range (2, row-2):
   for j in range (2, col-2):
       harmonic_mean_filter_image[i, j] = harmonic_filter(dup_img, i, j, 2, 2, 25)

print(dup_img)

for i in range (1, row-1):
   for j in range (1, col-1):
       geometric_mean_filter_image[i, j] = geometric_mean(dup_img, i, j, 1, 1, 9)
print(geometric_mean_filter_image)


plt.subplot(221)
plt.imshow(original_image, cmap='gray')
plt.title('original image')
plt.subplot(222)
plt.imshow(dup_img, cmap='gray')
plt.title('Noisy Image')
plt.subplot(223)
plt.imshow(harmonic_mean_filter_image, cmap='gray')
plt.title('harmonic mean filter')
plt.subplot(224)
plt.imshow(geometric_mean_filter_image, cmap='gray')
plt.title('Geometric Mean Filter')
plt.show()
