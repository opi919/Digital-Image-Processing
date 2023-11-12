import numpy as np
import cv2
import matplotlib.pyplot as plt


# Function to design a Butterworth low-pass filter
def butterworth_low_pass_filter(shape, cutoff_frequency, order):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    butterworth_filter = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            butterworth_filter[i, j] = 1 / (
                1 + (distance / cutoff_frequency) ** (2 * order)
            )

    return butterworth_filter


# Function to design a Gaussian low-pass filter
def gaussian_low_pass_filter(shape, cutoff_frequency):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    gaussian_filter = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            gaussian_filter[i, j] = np.exp(-0.5 * (distance / cutoff_frequency) ** 2)

    return gaussian_filter


# Read grayscale image
image = cv2.imread("glpf.png", cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noise = np.random.normal(0, 25, image.shape)
noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

# Fourier Transform
f_transform = np.fft.fft2(noisy_image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Apply Butterworth low-pass filter
cutoff_frequency_butterworth = 230
order_butterworth = 2
butterworth_filter = butterworth_low_pass_filter(
    image.shape, cutoff_frequency_butterworth, order_butterworth
)
filtered_image_butterworth = np.abs(
    np.fft.ifft2(f_transform_shifted * butterworth_filter)
)

# Apply Gaussian low-pass filter
cutoff_frequency_gaussian = 240
gaussian_filter = gaussian_low_pass_filter(image.shape, cutoff_frequency_gaussian)
filtered_image_gaussian = np.abs(np.fft.ifft2(f_transform_shifted * gaussian_filter))

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")

plt.subplot(2, 3, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")

plt.subplot(2, 3, 3)
plt.imshow(np.log(1 + np.abs(f_transform_shifted)), cmap="gray")
plt.title("Fourier Transform")

plt.subplot(2, 3, 4)
plt.imshow(butterworth_filter, cmap="gray")
plt.title("Butterworth Filter")

plt.subplot(2, 3, 5)
plt.imshow(filtered_image_butterworth, cmap="gray")
plt.title("Filtered Image (Butterworth)")

plt.subplot(2, 3, 6)
plt.imshow(filtered_image_gaussian, cmap="gray")
plt.title("Filtered Image (Gaussian)")

plt.show()
