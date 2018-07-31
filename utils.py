import os
import numpy as np
from PIL import Image



class Logger():
    def __init__(self, options):
        self.folder_path = options.gen_folder_path
        self.options = options
        self.make_folder()
        self.write_log()

    def write_log(self):
        file_path = self.folder_path + "config.txt"
        with open(file_path , "w") as f:
            option_dict = self.options.__dict__
            for key in option_dict:
                if key[0] != "_":
                    f.write("{}:\t\t{}\n".format(key, option_dict[key]))

    def make_folder(self):
        os.mkdir(self.folder_path)

    def save_img_step(self, step, gen_img, img_shape):
        img_path = self.folder_path + "step_" + str(step) + ".jpg"
        save_img(img_path, gen_img, img_shape)



def optimal_dimension(cont_img_path=None, styl_img_path=None, square=False):
    if cont_img_path is None and styl_img_path is None:
        return np.array([1, 224, 224, 3])
    height, width = Image.open(cont_img_path).size
    print(cont_img_height, cont_img_width)
    if square:
        max_len = max(cont_img_width, cont_img_height)
        return np.array([1, max_len, max_len, 3])
    return np.array([1, cont_img_height, cont_img_width, 3])

def load_img(path, shape=None, preprocess=False):
    image = Image.open(path)
    if shape is not None:
        shape = (shape[2], shape[1])
        image = image.resize(shape)
    img_array = np.asarray(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    if preprocess:
        img_array = rgb2bgr(img_array)
    print("Image loaded: ", path, "with dimension", shape)
    return img_array

def save_img(path, x, img_shape):
    img_array = np.copy(x)[0]
    #img_array = bgr2rgb(img_array)
    img_array = np.clip(img_array, 0, 255).astype('uint8')
    print(img_array)
    save_img = Image.fromarray(img_array)
    save_img.save(path, 'jpeg')
    print("Image saved as: ", path)
    return save_img

def white_img(img_shape):
    return  np.ones([1, img_shape[1], img_shape[2], 3]) * 255

def rand_img(img_shape):
    return np.random.uniform(0, 255, (1, img_shape[1], img_shape[2], 3))

def rgb2bgr(img_array):
    img_array[:, :, :, 0] -= 103.939
    img_array[:, :, :, 1] -= 116.779
    img_array[:, :, :, 2] -= 123.68
    img_array = img_array[:, :, :, ::-1]
    return img_array

def bgr2rgb(img_array):
    img_array = img_array[:, :, ::-1]
    img_array[:, :, 0] += 103.939
    img_array[:, :, 1] += 116.779
    img_array[:, :, 2] += 123.68
    return img_array
