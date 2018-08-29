import numpy as np
import tensorflow as tf

import utils
#from vgg.vgg import VGG19
from vgg.vgg_mat import VGG19

class StyleTransfer():
    def __init__(self, init_img, cont_img, styl_img, cont_layers, styl_layers, cont_weights, styl_weights, alpha, beta):
        self.init_img = init_img
        self.cont_img = cont_img
        self.styl_img = styl_img

        self.cont_layers = cont_layers
        self.styl_layers = styl_layers

        self.cont_weights = cont_weights
        self.styl_weights = styl_weights

        self.alpha = alpha
        self.beta = beta

        self.vgg = VGG19()
        self.graph = self.build_graph()
        self.step = 0

    def build_graph(self):
        with tf.Graph().as_default() as graph:
            with tf.variable_scope("inputs") as scope:
                self.image = tf.Variable(self.init_img, trainable=True, dtype=tf.float32)

            with tf.variable_scope("activitiy") as scope:
                styl_act = self.vgg.build(self.styl_img).layer_dict(self.styl_layers)
                self.styl_gram = {l: self._gram(styl_act[l]) for l in self.styl_layers}
                self.cont_act = self.vgg.build(self.cont_img).layer_dict(self.cont_layers)

                self.gen_styl_act = self.vgg.build(self.image).layer_dict(self.styl_layers)
                self.gen_cont_act = self.vgg.build(self.image).layer_dict(self.cont_layers)

            with tf.variable_scope("cont_loss") as scope:
                self.cont_loss = 0.
                for l in self.cont_layers:
                    self.cont_loss += self.cont_weights[l] * (1 / 2.) * tf.reduce_sum(tf.pow((self.gen_cont_act[l] - self.cont_act[l]), 2))

            with tf.variable_scope("styl_loss") as scope:
                self.styl_loss = 0.
                for l in self.styl_layers:
                    _, height, width, channels = self.gen_styl_act[l].get_shape()
                    feature_size = height.value * width.value
                    channels = channels.value
                    gen_styl_gram = self._gram(self.gen_styl_act[l])
                    self.styl_loss += self.styl_weights[l] * tf.reduce_sum(tf.pow((self.styl_gram[l] - gen_styl_gram), 2)) * (1 / (4 * feature_size**2 * channels**2))

            with tf.variable_scope("losses") as scope:
                self.total_loss = self.alpha * self.cont_loss + self.beta * self.styl_loss

            return graph

    def step_callback(self, img_shape, save_per_step):
        step_folder = "./step_folder/"
        def helper(image):
            if self.step % save_per_step == 0:
                image = np.reshape(image, img_shape)
                image = utils.img_postprocess(image)
                utils.save_image(step_folder, image, self.step)
                print("Image saved.")
        return helper

    def loss_callback(self):
        def helper(styl_loss, cont_loss, total_loss):
            print("Step: {}\tStyle Loss: {}\tContent Loss: {}\tTotal Loss: {}".format(self.step, styl_loss, cont_loss, total_loss))
            self.step += 1
        return helper


    def _gram(self, features):
        _, h, w, c = features.get_shape()
        features_t = tf.transpose(features, perm=(0, 3, 1, 2))
        matrix = tf.reshape(features_t, shape=[c.value, h.value*w.value])
        return tf.matmul(matrix, matrix, transpose_b=True)
