import cv2
import numpy as np

original_image = cv2.imread('image1.png')

window_size = (512, 512)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', *window_size)

cv2.imshow('Image', original_image)
cv2.waitKey(1000)

min_intensity = 0
max_intensity = 255

for num_bits in range(8, 0, -1):
    levels = 2 ** num_bits
    intensity_step = (max_intensity - min_intensity) / (levels - 1)
    quantized_image = np.floor_divide(original_image, intensity_step).astype(np.uint8) * intensity_step
    cv2.imshow('Image', cv2.resize(quantized_image, window_size))
    cv2.waitKey(1000)  # Display for 1 second

cv2.waitKey(0)
cv2.destroyAllWindows()
