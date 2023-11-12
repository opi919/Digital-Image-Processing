import cv2
import numpy as np
import matplotlib.pyplot as plt

original_image = cv2.imread("image2.jpg", 0)

row, col = original_image.shape

pwr = 5

power_image = original_image**pwr
inverse_log_image = np.log1p(original_image)  # Use np.log1p for numerical stability

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(inverse_log_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(power_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
