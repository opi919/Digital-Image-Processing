# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# # Function to design an ideal high-pass filter
# def ideal_high_pass_filter(shape, cutoff_frequency):
#     rows, cols = shape
#     center_row, center_col = rows // 2, cols // 2
#     ideal_filter = np.ones((rows, cols))

#     for i in range(rows):
#         for j in range(cols):
#             distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
#             if distance <= cutoff_frequency:
#                 ideal_filter[i, j] = 0

#     return ideal_filter

# # Function to design a Gaussian high-pass filter
# def gaussian_high_pass_filter(shape, cutoff_frequency):
#     rows, cols = shape
#     center_row, center_col = rows // 2, cols // 2
#     gaussian_filter = np.zeros((rows, cols))

#     for i in range(rows):
#         for j in range(cols):
#             distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
#             gaussian_filter[i, j] = 1 - np.exp(-0.5 * (distance / cutoff_frequency) ** 2)

#     return gaussian_filter

# # Read grayscale image
# image = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

# # Fourier Transform
# f_transform = np.fft.fft2(image)
# f_transform_shifted = np.fft.fftshift(f_transform)

# # Different radii for the high-pass filters
# cutoff_frequencies = [30, 50, 80]

# plt.figure(figsize=(12, 8))

# for i, cutoff_frequency in enumerate(cutoff_frequencies, 1):
#     # Apply ideal high-pass filter
#     ideal_filter = ideal_high_pass_filter(image.shape, cutoff_frequency)
#     filtered_f_transform_ideal = f_transform_shifted * ideal_filter
#     filtered_image_ideal = np.abs(np.fft.ifft2(filtered_f_transform_ideal))

#     # Apply Gaussian high-pass filter
#     gaussian_filter = gaussian_high_pass_filter(image.shape, cutoff_frequency)
#     filtered_f_transform_gaussian = f_transform_shifted * gaussian_filter
#     filtered_image_gaussian = np.abs(np.fft.ifft2(filtered_f_transform_gaussian))

#     # Display results
#     plt.subplot(3, len(cutoff_frequencies), i)
#     plt.imshow(image, cmap="gray")
#     plt.title("Original Image")

#     plt.subplot(3, len(cutoff_frequencies), i + len(cutoff_frequencies))
#     plt.imshow(filtered_image_ideal, cmap="gray")
#     plt.title(f"Ideal High-Pass (D0={cutoff_frequency})")

#     plt.subplot(3, len(cutoff_frequencies), i + 2 * len(cutoff_frequencies))
#     plt.imshow(filtered_image_gaussian, cmap="gray")
#     plt.title(f"Gaussian High-Pass (D0={cutoff_frequency})")

# plt.tight_layout()
# plt.show()


import numpy as np
import cv2
import matplotlib.pyplot as plt


# Function to design an ideal high-pass filter
def ideal_high_pass_filter(shape, cutoff_frequency):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    ideal_filter = np.ones((rows, cols))

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance <= cutoff_frequency:
                ideal_filter[i, j] = 0

    return ideal_filter


# Function to design a Gaussian high-pass filter
def gaussian_high_pass_filter(shape, cutoff_frequency):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    gaussian_filter = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            gaussian_filter[i, j] = 1 - np.exp(
                -0.5 * (distance / cutoff_frequency) ** 2
            )

    return gaussian_filter


# Read grayscale image
image = cv2.imread("image1.png", cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noise = np.random.normal(0, 25, image.shape)
noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

# Fourier Transform
f_transform = np.fft.fft2(noisy_image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Cutoff frequency for high-pass filters
cutoff_frequency = 30

# Apply ideal high-pass filter to the noisy image for edge detection
ideal_filter = ideal_high_pass_filter(image.shape, cutoff_frequency)
edges_ideal_noisy = np.abs(np.fft.ifft2(f_transform_shifted * ideal_filter))

# Apply Gaussian high-pass filter to the noisy image for edge detection
gaussian_filter = gaussian_high_pass_filter(image.shape, cutoff_frequency)
edges_gaussian_noisy = np.abs(np.fft.ifft2(f_transform_shifted * gaussian_filter))

# Display results
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")

plt.subplot(2, 3, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")

plt.subplot(2, 3, 3)
plt.imshow(np.log(1 + np.abs(f_transform_shifted)), cmap="gray")
plt.title("Fourier Transform (Noisy)")

plt.subplot(2, 3, 4)
plt.imshow(edges_ideal_noisy, cmap="gray")
plt.title("Edges (Ideal High-Pass)")

plt.subplot(2, 3, 5)
plt.imshow(edges_gaussian_noisy, cmap="gray")
plt.title("Edges (Gaussian High-Pass)")

plt.tight_layout()
plt.show()
