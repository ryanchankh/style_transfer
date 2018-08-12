from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

from style_transfer2 import StyleTransfer
import utils


class OPTIONS():

    init_time = datetime.now().strftime("%H%M%S_%Y%m%d")

    # image paths
    styl_img_path = './images/style/starry_night.jpg'
    cont_img_path = './images/content/tubingen.jpg'
    gen_folder_path = "./gen_img/" + init_time + "/"

    # image shape  (Requirement: (batch, height, width, channels))
    #img_shape = utils.optimal_dimension(cont_img_path, styl_img_path, square=False)
    #img_shape = utils.optimal_dimension() # [batch, width, height, channels]
    img_shape = np.array([1, 512, 512, 3])
    init_img = "styl" #"rand_uni"

    # hyper-parameters
    alpha = 1e-3           # style weight alpha
    beta = 1         # content weight beta
    num_steps = 1000     # training iterations
    save_per_step = 30   # save image per this number of steinitializer

   # content and style layers used in style transfer
    cont_layers = ["conv4_2"]
    styl_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]


    # weights on each style layer
    styl_weights = {"relu1_1": 0.2, "relu2_1": 0.2, "relu3_1": 0.2, "relu4_1": 0.2, "relu5_1": 0.2}

def main():

    logger = utils.Logger(OPTIONS)

    # load images into numpy arrays
    styl_img = utils.load_img(OPTIONS.styl_img_path, shape=OPTIONS.img_shape, preprocess=True)
    cont_img = utils.load_img(OPTIONS.cont_img_path, shape=OPTIONS.img_shape, preprocess=True)

    # initialize style transfer model
    model = StyleTransfer(OPTIONS.init_img,
                          styl_img,
                          cont_img,
                          OPTIONS.img_shape,
                          OPTIONS.cont_layers,
                          OPTIONS.styl_layers,
                          OPTIONS.styl_weights,
                          OPTIONS.alpha,
                          OPTIONS.beta)

    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())

        optimizer = ScipyOptimizerInterface(model.total_loss, options={'maxiter': OPTIONS.num_steps})
        optimizer.minimize(sess,
                           feed_dict={},
                           fetches=[model.styl_loss, model.cont_loss, model.total_loss],
                           step_callback=model.step_callback(logger, sess, OPTIONS.save_per_step),
                           loss_callback=model.loss_callback())
        gen_img = sess.run(model.img)
        utils.save_img(OPTIONS.gen_folder_path+"result.jpg", gen_img, OPTIONS.img_shape)
        print("Style Transfer Complete.")




if __name__ == "__main__":
    main()
