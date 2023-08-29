import cv2

# Load the image
img = cv2.imread('image1.jpg')

# Set the new resolution
new_width, new_height = (560, 560)

# Resize the image
img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Save the resized image
cv2.imwrite('resized_image.jpg', img_resized)

# Half the resolution
half_width, half_height = (new_width // 2, new_height // 2)

# Resize the image again
img_half_resized = cv2.resize(img_resized, (half_width, half_height), interpolation=cv2.INTER_AREA)

# Save the half-resized image
cv2.imwrite('half_resized_image.jpg', img_half_resized)

# Set the window size
window_width, window_height = (560, 560)

# Create named windows
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Half Resolution', cv2.WINDOW_NORMAL)

# Set the window sizes
cv2.resizeWindow('Original', window_width, window_height)
cv2.resizeWindow('Half Resolution', window_width, window_height)

# Display the images
cv2.imshow('Original', img_resized)
cv2.imshow('Half Resolution', img_half_resized)

# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
