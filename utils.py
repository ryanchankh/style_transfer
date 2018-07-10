"""
Utilities.
Date: 17 June 2018
Author: Ryan Chan
"""

import numpy as np
from PIL import Image

def optimal_dimension(cont_img_path, styl_img_path):
    cont_img_width, cont_img_height = Image.open(cont_img_path).size
    return np.array([1, cont_img_width, cont_img_height, 3])

def load_image(path, shape=None):
    image = Image.open(path)
    if shape is not None:
        shape = (shape[1], shape[2])
        image = image.resize(shape)
    img_array = np.asarray(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array[:, :, :, 0] -= 103.939
    img_array[:, :, :, 1] -= 116.779
    img_array[:, :, :, 2] -= 123.68
    img_array = img_array[:, :, :, ::-1]

    print("Image loaded: ", path, "with dimension", shape)
    return img_array

def save_image(path, x, img_shape):
    img_array = np.copy(x)
    img_array = img_array.reshape((img_shape[1], img_shape[2], 3))

    img_array = img_array[:, :, ::-1]
    img_array[:, :, 0] += 103.939
    img_array[:, :, 1] += 116.779
    img_array[:, :, 2] += 123.68

    img_array = np.clip(img_array, 0, 255).astype('uint8')
    img_arrays = np.copy(img_array)[:, :, ::-1]
    save_img = Image.fromarray(img_array)
    save_img.save(path)
    print("Image saved as: ", path)
    return save_img

