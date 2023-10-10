import cv2
import numpy as np

# Load the original grayscale image of size 512x512
original_image = cv2.imread('image1.png' )

# Determine the window size
window_size = (512, 512)

# Create a window to display images
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', *window_size)

# Display the original image
cv2.imshow('Image', original_image)
cv2.waitKey(0)

# Reduce intensity resolution to binary
intensity_step = 255 / 1  # Binary format (1-bit intensity)
quantized_image = (original_image // intensity_step) * intensity_step

# Convert to binary image
binary_image = np.where(quantized_image >= 128, 255, 0).astype(np.uint8)

# Display binary image
cv2.imshow('Image', cv2.resize(binary_image, window_size))
cv2.waitKey(0)

cv2.destroyAllWindows()
