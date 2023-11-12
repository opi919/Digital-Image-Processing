import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image in grayscale
image = cv2.imread("image1.png", 0)

# Calculate the histogram
hist = np.zeros(256)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        hist[image[i, j]] += 1

# Display the original image and its histogram
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Histogram
plt.subplot(1, 4, 2)
plt.plot(hist, color="black")
plt.title("Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

# Perform single threshold segmentation
threshold_value = 128  # Adjust this threshold as needed
segmented_image = (image > threshold_value) * 255

# Display the segmented image and its histogram
plt.subplot(1, 4, 3)
plt.imshow(segmented_image, cmap="gray")
plt.title("Segmented Image")
plt.axis("off")

segmented_hist = np.zeros(256)

for i in range(segmented_image.shape[0]):
    for j in range(segmented_image.shape[1]):
        segmented_hist[segmented_image[i, j]] += 1

plt.subplot(1, 4, 4)
plt.plot(segmented_hist, color="black")
plt.title("Histogram after Segmentation")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.show()
