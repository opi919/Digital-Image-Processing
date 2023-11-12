import cv2
import numpy as np

original_image = cv2.imread('image1.png')

window_size = (512, 512)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', *window_size)

cv2.imshow('Image', original_image)
cv2.waitKey(1000) 

resized_image = original_image.copy()
while resized_image.shape[0] > 10 and resized_image.shape[1] > 10:
    resized_image = cv2.resize(resized_image, (resized_image.shape[0] // 2, resized_image.shape[1] // 2))
    resized_display = cv2.resize(resized_image, window_size)
    cv2.imshow('Image', resized_display)
    cv2.waitKey(1000)  

cv2.waitKey(0)
cv2.destroyAllWindows()
