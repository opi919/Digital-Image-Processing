import numpy as np


# Function to compute the 2D Fourier transform manually
def manual_fft2(image):
    rows, cols = image.shape
    f_transform = np.zeros((rows, cols), dtype=np.complex128)

    for u in range(rows):
        for v in range(cols):
            sum_val = 0
            for x in range(rows):
                for y in range(cols):
                    sum_val += image[x, y] * np.exp(
                        -2j * np.pi * ((u * x) / rows + (v * y) / cols)
                    )
            f_transform[u, v] = sum_val

    return f_transform


# Function to compute the 2D inverse Fourier transform manually
def manual_ifft2(f_transform):
    rows, cols = f_transform.shape
    inv_f_transform = np.zeros((rows, cols), dtype=np.complex128)

    for x in range(rows):
        for y in range(cols):
            sum_val = 0
            for u in range(rows):
                for v in range(cols):
                    sum_val += f_transform[u, v] * np.exp(
                        2j * np.pi * ((u * x) / rows + (v * y) / cols)
                    )
            inv_f_transform[x, y] = sum_val / (rows * cols)

    return inv_f_transform
