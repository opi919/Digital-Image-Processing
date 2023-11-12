import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
original_image = cv2.imread("image1.png", cv2.IMREAD_GRAYSCALE)

# Extract the last three bits (MSB) of each pixel
last_three_bits_image = original_image & 0b00000111

# Create the difference image
difference_image = np.abs(original_image - last_three_bits_image)

# Display the original image, last three bits image, and the difference image
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Last Three Bits Image
plt.subplot(1, 3, 2)
plt.imshow(last_three_bits_image, cmap="gray")
plt.title("Last Three Bits Image (MSB)")
plt.axis("off")

# Difference Image
plt.subplot(1, 3, 3)
plt.imshow(difference_image, cmap="gray")
plt.title("Difference Image")
plt.axis("off")

plt.show()
