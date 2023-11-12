import cv2
import numpy as np
import matplotlib.pyplot as plt

original_image = cv2.imread("image1.png",cv2.IMREAD_GRAYSCALE)

window_size = (512, 512)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", *window_size)

cv2.imshow("Image", original_image)
cv2.waitKey(1000)

min_intensity = 0
max_intensity = 255

quantized_image = []
for num_bits in range(1, 9):
    levels = 2**num_bits - 1
    quantization_step = (max_intensity - min_intensity) / (levels)
    quantized_image = (original_image / quantization_step).astype(
        np.uint8
    ) * quantization_step
    quantized_image.append(quantized_image)
    cv2.imshow("Image", cv2.resize(quantized_image, window_size))
    cv2.waitKey(1000)

# conver to binary(1bit)
binary_image = cv2.convertScaleAbs(np.where(original_image > 127, 255, 0))
binary_display = cv2.resize(binary_image, window_size)
cv2.imshow("Image", binary_display)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
