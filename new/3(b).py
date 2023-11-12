import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to add salt and pepper noise to an image
def add_salt_and_pepper_noise(image):
    salt_prob = 0.01
    pepper_prob = 0.01
    noisy_image = image.copy()
    total_pixels = image.size

    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# Function to apply average filter
def apply_average_filter(image, kernel_size):
    rows, cols = image.shape
    result = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            sum_val = 0
            count = 0
            for m in range(-kernel_size // 2, kernel_size // 2 + 1):
                for n in range(-kernel_size // 2, kernel_size // 2 + 1):
                    if 0 <= i + m < rows and 0 <= j + n < cols:
                        sum_val += image[i + m, j + n]
                        count += 1

            result[i, j] = sum_val / count

    return result

plt.figure(figsize=(10, 10))

original_image = cv2.imread("image1.png", 0)
plt.subplot(3, 3, 1)
plt.imshow(original_image, cmap="gray")
plt.title("Original Image")

noisy_image = add_salt_and_pepper_noise(original_image)
plt.subplot(3, 3, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")

kernel_sizes = [3, 5, 7]
for kernel_size in kernel_sizes:
    average_filtered_image = apply_average_filter(noisy_image, kernel_size)

    plt.subplot(3, 3, kernel_sizes.index(kernel_size) + 3)
    plt.imshow(average_filtered_image, cmap="gray")
    plt.title(f"Average Filter ({kernel_size}x{kernel_size})")

plt.tight_layout()
plt.show()
