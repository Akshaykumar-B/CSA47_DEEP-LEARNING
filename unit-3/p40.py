"""Compare max pooling and average pooling on the same image features."""

import numpy as np


def pool2d(image, mode, size=2):
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or image.shape[0] % size or image.shape[1] % size:
        raise ValueError("image dimensions must be divisible by the pooling window size")
    output = np.empty((image.shape[0] // size, image.shape[1] // size))
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            block = image[row * size:(row + 1) * size, column * size:(column + 1) * size]
            if mode == "max":
                output[row, column] = block.max()
            elif mode == "average":
                output[row, column] = block.mean()
            else:
                raise ValueError("mode must be 'max' or 'average'")
    return output


feature_map = np.array([
    [0, 1, 2, 1, 0, 0],
    [1, 5, 3, 0, 1, 0],
    [0, 2, 1, 4, 3, 1],
    [0, 0, 2, 3, 1, 0],
    [1, 0, 0, 2, 5, 4],
    [0, 1, 1, 0, 3, 2],
], dtype=float)
print("Input feature map:\n", feature_map)
print("Max pooling:\n", pool2d(feature_map, "max"))
print("Average pooling:\n", pool2d(feature_map, "average"))
print("Max pooling preserves strong activations; average pooling summarizes local intensity.")
