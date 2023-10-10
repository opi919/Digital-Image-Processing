import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image2.jpg",0)

sobel_operator = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
print(sobel_operator)

new_image = cv2.filter2D(img,-1,sobel_operator)

plt.subplot(2,1,1)
plt.imshow(img,cmap='gray')

plt.subplot(2,1,2)
plt.imshow(new_image,cmap='gray')

plt.show()
