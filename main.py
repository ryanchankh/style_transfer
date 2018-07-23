from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

from style_transfer import StyleTransfer
import utils

class OPTIONS():


    init_time = datetime.now().strftime("%H%M%S_%Y%m%d")

    # image paths
    styl_img_path = './images/style/starry_night.jpg'
    cont_img_path = './images/content/tubingen.jpg'
    white_img_path = './images/others/plain_white.jpg'
    gen_folder_path = "./gen_img/" + init_time + "/"

    # hyper-parameters
    #img_shape = utils.optimal_dimension(cont_img_path, styl_img_path, square=False)
    #img_shape = utils.optimal_dimension() # [batch, width, height, channels]
    img_shape = np.array([1, 100, 300, 3])
    alpha = 5           # style weight alpha
    beta = 0.025         # content weight beta
    l_rate = 0
    num_steps = 10     # training iterations
    save_per_step = 5   # save image per this number of step

   # content and style layers used in style transfer
    cont_layers = ["conv4_2"]
    styl_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    # weights on each style layer
    styl_weights = {"conv1_1": 0.2, "conv1_2": 0.2, "pool1": 0,
                    "conv2_1": 0.2, "conv2_2": 0, "pool2": 0,
                    "conv3_1": 0.2, "conv3_2": 0, "conv3_3": 0.2, "conv3_4": 0, "pool3": 0,
                    "conv4_1": 0.2, "conv4_2": 0, "conv4_3": 0.2, "conv4_4": 0, "pool4": 0,
                    "conv5_1": 0.2, "conv5_2": 0, "conv5_3": 0.2, "conv5_4": 0, "pool5": 0}



logger = utils.Logger(OPTIONS)

# turn image into numpy arrays
styl_img = utils.load_img(OPTIONS.styl_img_path, shape=OPTIONS.img_shape)
cont_img = utils.load_img(OPTIONS.cont_img_path, shape=OPTIONS.img_shape)
init_img = utils.load_img(OPTIONS.white_img_path, shape=OPTIONS.img_shape)
white_img = utils.white_img(OPTIONS.img_shape)
rand_img = utils.rand_img(OPTIONS.img_shape)

model = StyleTransfer(rand_img,
                      OPTIONS.img_shape,
                      OPTIONS.cont_layers,
                      OPTIONS.styl_layers,
                      OPTIONS.styl_weights,
                      OPTIONS.alpha,
                      OPTIONS.beta,
                      OPTIONS.l_rate)


with tf.Session(graph=model.graph) as sess:

    sess.run(tf.global_variables_initializer())
    feed_dict = {model.cont_img: cont_img, model.styl_img: styl_img}

    optimizer = ScipyOptimizerInterface(model.total_loss, options={'maxiter': OPTIONS.num_steps})
    optimizer.minimize(sess,
                       feed_dict=feed_dict,
                       step_callback=model.step_callback(logger, sess, OPTIONS.save_per_step))
    gen_img = sess.run(model.image)
    utils.save_img(OPTIONS.gen_folder_path+"result.jpg", gen_img, OPTIONS.img_shape)
