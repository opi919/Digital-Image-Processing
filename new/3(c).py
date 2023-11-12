import numpy as np
import cv2
import matplotlib.pyplot as plt


# Function to apply harmonic mean filter
def apply_harmonic_mean_filter(image, kernel_size):
    rows, cols = image.shape
    result = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            values = []
            for m in range(-kernel_size // 2, kernel_size // 2 + 1):
                for n in range(-kernel_size // 2, kernel_size // 2 + 1):
                    if (
                        0 <= i + m < rows
                        and 0 <= j + n < cols
                        and image[i + m, j + n] != 0
                    ):
                        values.append(1 / image[i + m, j + n])

            if len(values) > 0:
                # Harmonic mean formula
                # result[i, j] = N / (1/image[i-m, j-n] + 1/image[i+m, j+n] + ...)
                result[i, j] = len(values) / np.sum(values)
            else:
                result[i, j] = 0

    return 1 / result


# Function to apply geometric mean filter
def apply_geometric_mean_filter(image, kernel_size):
    rows, cols = image.shape
    result = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            values = []
            for m in range(-kernel_size // 2, kernel_size // 2 + 1):
                for n in range(-kernel_size // 2, kernel_size // 2 + 1):
                    if (
                        0 <= i + m < rows
                        and 0 <= j + n < cols
                        and image[i + m, j + n] != 0
                    ):
                        values.append(np.log(image[i + m, j + n]))

            if len(values) > 0:
                # Geometric mean formula
                # result[i, j] = exp((1/num_pixels) * sum(log(image[i+m, j+n])))
                result[i, j] = np.exp(np.sum(values) / len(values))
            else:
                result[i, j] = 0

    return result


# Load the noisy image
noisy_image = cv2.imread("noisy_image.jpg", cv2.IMREAD_GRAYSCALE)

# Display the noisy image
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")

# Apply harmonic mean filter
harmonic_mean_filtered_image = apply_harmonic_mean_filter(noisy_image, 3)
plt.subplot(1, 4, 2)
plt.imshow(harmonic_mean_filtered_image, cmap="gray")
plt.title("Harmonic Mean Filter")

# Apply geometric mean filter
geometric_mean_filtered_image = apply_geometric_mean_filter(noisy_image, 3)
plt.subplot(1, 4, 3)
plt.imshow(geometric_mean_filtered_image, cmap="gray")
plt.title("Geometric Mean Filter")
plt.tight_layout()
plt.show()
