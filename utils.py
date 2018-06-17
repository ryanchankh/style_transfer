"""
Utilities. 
Date: 17 June 2018
Author: Ryan Chan
"""

import numpy as np
from PIL import Image

def optimal_dimension(cont_img_path, styl_img_path):
    """
    Find optimal dimension for clearest content image.
    """

    cont_img_width, cont_img_height = Image.open(cont_img_path).size
    return np.array([1, cont_img_width, cont_img_height, 3])

def load_image(path, shape=None):
    """
    Turn image into numpy array. Also subtract all three channels by 
    mean pixel values for better outcome. 
    """

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

def save_image(path, img_array):
    """
    Save image. 
    """

    image = np.copy(img_array)
    save_img = show_img(image)
    save_img.save(path)
    print("Image saved as: ", path)
    return save_img
    
def show_img(array):
    """
    Turn numpy array back into image. Also re-add all three channels by mean pixel 
    values. 
    """

    img_array = np.copy(array[0])
    img_array = img_array[:, :, ::-1]
    img_array[:, :, 0] += 103.939
    img_array[:, :, 1] += 116.779
    img_array[:, :, 2] += 123.68
    return Image.fromarray(np.clip(img_array, 0, 255).astype('uint8'))
