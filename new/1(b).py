import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image1.png", 0)


# Function to decrease intensity level resolution by one bit
def decrease_resolution(image, num_bits):
    # Calculate the shift value to decrease intensity resolution
    shift_value = 8 - num_bits

    # Right shift the pixel values to discard lower bits, reducing intensity resolution
    shifted_right = image >> shift_value

    # Left shift the pixel values back to their original position, filling with zeros
    # This effectively decreases intensity resolution by zeroing out lower bits
    decreased_resolution_image = shifted_right << shift_value

    return decreased_resolution_image


# Decrease intensity level resolution by one bit at a time and display the images
reduced_images = []
for i in range(0, 8, 1):
    reduced_image = decrease_resolution(image, i)
    reduced_images.append(reduced_image)


fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle("Decreased intesity resolution", fontsize=16)

for i, ax in enumerate(axes.flat):
    ax.imshow(reduced_images[i], cmap="gray")
    ax.set_title(f"{i+1} bits")
    ax.axis("off")

plt.tight_layout()
plt.show()
