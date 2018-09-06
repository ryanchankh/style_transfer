from datetime import datetime

import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf

def load_image(path, img_shape=None):
    img_array = skimage.io.imread(path)

    if img_shape is not None:
        img_shape = img_shape[1:3]
        img_array = skimage.transform.resize(img_array, img_shape, mode="constant")
    else:
        img_array = img_array / 255.
    img_array = np.float32(img_array) 
    return img_array

def load_init_image(cont_img, styl_img, img_shape, choice=""):
    if choice == "cont":
        return cont_img.copy()
    elif choice == "styl":
        return styl_img.copy()
    elif choice == "white":
        return np.zeros(img_shape[1:], dtype=np.float32) + 255
    else:
        return np.float32(np.random.uniform(0, 1, size=img_shape[1:]))

def optimal_dimension(cont_path, square=False):
    full_shape = skimage.io.imread(cont_path).shape
    height, width, channel = full_shape[0], full_shape[1], full_shape[2]

    if square:
        max_len = max(height, width)
        return (1, max_len, max_len, channel)
    return (1, height, width, channel)

def save_image(path, img_array, step=None):
    img_array = np.clip(img_array*255., 0, 255).astype('uint8')

    if step is None:
        file_name = path + datetime.now().strftime("%H%M%S_%Y%m%d") + ".jpg"
    else:
        file_name = path + datetime.now().strftime("%H%M%S_%Y%m%d") + "step" + str(step) + ".jpg"

    skimage.io.imsave(file_name, img_array)

def img_preprocess(img_array):
    # add batch dimension to image
    img_array = np.expand_dims(img_array, 0)

    # subtract mean pixel values
    img_array[:, :, :, 0] -= 103.939 / 255.
    img_array[:, :, :, 1] -= 116.779 / 255.
    img_array[:, :, :, 2] -= 123.68 / 255.

    # convert RGB to BGR
    img_array = img_array[:, :, :, ::-1]

    return img_array

def img_postprocess(img_array):
    # make copy of reference array and reduce dimension
    img_array = np.copy(img_array[0])

    # convert back from BGR to RGB
    img_array = img_array[:, :, ::-1]

    # add mean pixel values to img_array
    img_array[:, :, 0] += 103.939 / 255.
    img_array[:, :, 1] += 116.779 / 255.
    img_array[:, :, 2] += 123.68 / 255.

    return img_array
