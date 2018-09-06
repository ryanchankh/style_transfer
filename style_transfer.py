import numpy as np
import tensorflow as tf

import utils
from vgg.vgg import VGG19
#from vgg.vgg_mat import VGG19

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
            with tf.name_scope("inputs") as scope:
                self.image = tf.Variable(self.init_img, trainable=True, dtype=tf.float32)

            with tf.name_scope("activitiy") as scope:
                self.styl_act = self.vgg.build(self.styl_img).layer_dict(self.styl_layers)
                self.styl_gram = {l: self._gram(self.styl_act[l]) for l in self.styl_layers}
                #self.styl_gram2 = {l: self.naive_gram(self.styl_act[l]) for l in self.styl_layers}
                self.cont_act = self.vgg.build(self.cont_img).layer_dict(self.cont_layers)

                img_model = self.vgg.build(self.image)
                self.gen_cont_act = VGG19().layer_dict(self.cont_layers)
                self.gen_styl_act = .layer_dict(self.styl_layers)

            with tf.name_scope("cont_loss") as scope:
                self.cont_loss = 0.
                self.cont_loss_list = []
                for l in self.cont_layers:
                    P = self.cont_act[l]
                    F = self.gen_cont_act[l]
                    w = self.cont_weights[l]
                    layer_loss = w * (1. / 2.) * tf.reduce_sum(tf.pow((F - P), 2))
                    self.cont_loss_list.append(layer_loss)
                    self.cont_loss += layer_loss

            with tf.name_scope("styl_loss") as scope:
                self.styl_loss = 0.
                self.styl_loss_list = []
                for l in self.styl_layers:
                    _, h, w, c = self.gen_styl_act[l].get_shape()
                    M = h.value * w.value
                    N = c.value
                    A = self.styl_gram[l]
                    G = self._gram(self.gen_styl_act[l])
                    w = self.styl_weights[l]
                    layer_loss = w * (1. / (4. * M**2 * N**2)) * tf.reduce_sum(tf.pow((G - A), 2))
                    self.styl_loss_list.append(layer_loss)
                    self.styl_loss += layer_loss

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
        def helper(styl_loss, cont_loss, total_loss, styl_loss_list, cont_loss_list, gen_cont_act, gen_styl_act, image):
            print("Step: {}\tStyle Loss: {}\tContent Loss: {}\tTotal Loss: {}".format(self.step, styl_loss, cont_loss, total_loss))
            print("styl_loss_list: {}".format(styl_loss_list))
            print("cont_loss_list: {}".format(cont_loss_list))
            #print("gen_cont_act: {}".format(gen_cont_act[self.cont_layers[0]]))
            #print("cont_act: {}".format(cont_act[self.cont_layers[0]]))
            #print("gen_styl_act")
            #[print(l, gen_styl_act[l]) for l in self.styl_layers]
            print()
            self.step += 1
        return helper


    def _gram(self, features):
        _, h, w, c = features.get_shape()
        features_t = tf.transpose(features, perm=(0, 3, 1, 2))
        matrix = tf.reshape(features_t, shape=[c.value, h.value*w.value])
        return tf.matmul(matrix, matrix, transpose_b=True)

    def naive_gram(self, features):
        _, h, w, c = features.get_shape()
        gram = []
        for i in np.arange(c.value):
            for j in np.arange(c.value):
                Fi = tf.reshape(features[0, :, :, i], shape=[-1])
                Fj = tf.reshape(features[0, :, :, j], shape=[-1])
                gram_ij = tf.tensordot(Fi, Fj, 1)
                gram.append(gram_ij)
        return tf.reshape(gram, (c.value, c.value))
