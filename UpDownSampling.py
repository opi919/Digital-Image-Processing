import cv2

# Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform upsampling using bicubic interpolation
upsampled_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Perform downsampling by taking every 2nd pixel
downsampled_image = image[::2, ::2]

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Upsampled Image', upsampled_image)
cv2.imshow('Downsampled Image', downsampled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
