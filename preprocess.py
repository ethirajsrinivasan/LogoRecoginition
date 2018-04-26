PIXEL_DEPTH = 255.0

import numpy as np


def scaling(image_data):
    return (image_data.astype(np.float32) - (PIXEL_DEPTH / 2)) / PIXEL_DEPTH