import numpy as np
import cv2
import matplotlib.pyplot as plt


# Function to design an ideal low-pass filter
def ideal_low_pass_filter(shape, cutoff_frequency):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    ideal_filter = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance <= cutoff_frequency:
                ideal_filter[i, j] = 1

    return ideal_filter


# Read grayscale image
image = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noise = np.random.normal(0, 25, image.shape)
noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

# Fourier Transform
f_transform = np.fft.fft2(noisy_image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Different radii for the ideal low-pass filter
cutoff_frequencies = [30, 50, 80]

plt.figure(figsize=(12, 8))

for i, cutoff_frequency in enumerate(cutoff_frequencies, 1):
    # Apply ideal low-pass filter
    ideal_filter = ideal_low_pass_filter(image.shape, cutoff_frequency)
    filtered_image = np.abs(np.fft.ifft2(f_transform_shifted * ideal_filter))

    # Display results
    plt.subplot(2, len(cutoff_frequencies), i)
    plt.imshow(filtered_image, cmap="gray")
    plt.title(f"Cutoff Frequency = {cutoff_frequency}")

    # Display the corresponding frequency domain representation
    plt.subplot(2, len(cutoff_frequencies), i + len(cutoff_frequencies))
    plt.imshow(np.log(1 + np.abs(f_transform_shifted * ideal_filter)), cmap="gray")
    plt.title("Frequency Domain")

plt.show()
