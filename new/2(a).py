import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread("image1.png", cv2.IMREAD_GRAYSCALE)

# Define the range of gray levels for enhancement
lower_limit = 100  # Adjust as needed
upper_limit = 200  # Adjust as needed

# Perform brightness enhancement on the specified range of gray levels
enhanced_image = image.copy()
enhanced_image[
    (image >= lower_limit) & (image <= upper_limit)
] += 50  # Adjust the enhancement factor as needed

# Display the original and enhanced images
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Enhanced Image
plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap="gray")
plt.title("Enhanced Image")
plt.axis("off")

plt.show()
