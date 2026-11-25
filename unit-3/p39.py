"""Apply a 2-D convolution and max pooling to a small surveillance-style image."""

import numpy as np


def convolve2d(image, kernel):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be two-dimensional")
    if kernel.shape[0] > image.shape[0] or kernel.shape[1] > image.shape[1]:
        raise ValueError("kernel must fit inside the image")
    height = image.shape[0] - kernel.shape[0] + 1
    width = image.shape[1] - kernel.shape[1] + 1
    result = np.empty((height, width))
    for row in range(height):
        for column in range(width):
            region = image[row:row + kernel.shape[0], column:column + kernel.shape[1]]
            result[row, column] = np.sum(region * kernel)
    return result


def max_pool2d(feature_map, size=2):
    height = feature_map.shape[0] // size
    width = feature_map.shape[1] // size
    pooled = np.empty((height, width))
    for row in range(height):
        for column in range(width):
            block = feature_map[row * size:(row + 1) * size, column * size:(column + 1) * size]
            pooled[row, column] = block.max()
    return pooled


image = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 3, 3, 1, 0],
    [0, 1, 3, 3, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=float)
edge_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
features = convolve2d(image, edge_filter)
print("Input image:\n", image)
print("Edge feature map:\n", features)
print("Max-pooled feature map:\n", max_pool2d(features))
print("Convolution highlights vertical intensity changes; pooling reduces spatial resolution.")
