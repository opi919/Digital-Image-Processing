import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("image1.png")

# Set up the subplot grid
num_rows, num_cols = 2, 4
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))

# Resize and store images in a list
resized_images = []
resized_image = original_image.copy()
resized_images.append(resized_image)
while resized_image.shape[0] > 10 and resized_image.shape[1] > 10:
    resized_image = cv2.resize(
        resized_image, (resized_image.shape[0] // 2, resized_image.shape[1] // 2)
    )
    resized_images.append(resized_image)

# Display all resized images in the same window
for i, img in enumerate(resized_images):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(img, cmap="gray")
    axes[row, col].set_title(f"Resized {img.shape[0]}x{img.shape[1]}")
    axes[row, col].axis("off")

# Show the final plot
plt.tight_layout()
plt.show()
