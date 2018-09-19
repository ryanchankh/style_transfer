import json
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils
#from vgg.vgg import VGG19
from vgg.vgg_mat import VGG19

class StyleTransfer():
    """Style Transfer Model.

    @param:
        init_ing: Initial Image. Training initialized with this image.
        cont_img: Content Image.
        styl_img: Style Image.
        cont_layers: Layers chosen for content activity.
        styl_layers: Layers chose for style activity.
        cont_weights: Weights for each content activity layer.
        styl_weights: Weights for each style activity layer.
        alpha: learning rate/trade-off for content loss.
        beta: learning rate/trade-off for styl_loss.
    """

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
        """Tensorflow graph. Precomputes content and style activity for training."""

        with tf.Graph().as_default() as graph:
            with tf.name_scope("inputs") as scope:
                self.image = tf.Variable(self.init_img, trainable=True, dtype=tf.float32)

            with tf.name_scope("activitiy") as scope:
                self.styl_act = self.vgg.build(self.styl_img).layer_dict(self.styl_layers)
                self.styl_gram = {l: self._gram(self.styl_act[l]) for l in self.styl_layers}
                self.cont_act = self.vgg.build(self.cont_img).layer_dict(self.cont_layers)

                img_model = self.vgg.build(self.image)
                self.gen_styl_act = img_model.layer_dict(self.styl_layers)
                self.gen_cont_act = img_model.layer_dict(self.cont_layers)

            with tf.name_scope("cont_loss") as scope:
                self.cont_loss = 0.
                self.cont_loss_list = []
                for l in self.cont_layers:
                    P = self.cont_act[l]
                    F = self.gen_cont_act[l]
                    w = self.cont_weights[l]
#                    cont_layer_loss = w * tf.reduce_sum(tf.pow((F - P), 2)) / 2
                    cont_layer_loss = w * (2 * tf.nn.l2_loss(F - P) / tf.size(P, out_type=tf.float32))
                    self.cont_loss_list.append(cont_layer_loss)
                    self.cont_loss += cont_layer_loss

            with tf.name_scope("styl_loss") as scope:
                self.styl_loss = 0.
                self.styl_loss_list = []
                for l in self.styl_layers:
                    _, h, w, c = self.gen_styl_act[l].get_shape()
                    M = h.value * w.value
                    N = c.value
                    A = self.styl_gram[l]
                    G = self._gram(self.gen_styl_act[l])
                    lw = self.styl_weights[l]
#                    styl_layer_loss = lw / (4. * M**2 * N**2) * tf.reduce_sum(tf.pow((G - A), 2))
                    styl_layer_loss = lw * (2 * tf.nn.l2_loss(G - A) / tf.size(G, out_type=tf.float32) )
                    self.styl_loss_list.append(styl_layer_loss)
                    self.styl_loss += styl_layer_loss

            with tf.name_scope("losses") as scope:
                self.total_loss = self.alpha * self.cont_loss + self.beta * self.styl_loss

            return graph

    def step_callback(self, img_shape, save_per_step):
        """Optimizer member function. Higher order function called from main.py"""

        step_folder = "./step_folder/"
        def helper(image):
            if self.step % save_per_step == 0:
                image = np.reshape(image, img_shape)
                image = utils.img_postprocess(image)
                utils.save_image(step_folder, image, self.step)
                print("Image saved.")

            if self.step % 500 == 0:
                filename = "./step_folder/loss_" + str(self.step) + "_" + datetime.now().strftime("%H%M%S_%Y%m%d")  +".log"
                with open(filename, "w+") as file:
                    string = json.dumps(self.loss_track, indent=2)
                    file.write(string)
        return helper

    def loss_callback(self):
        """Optimizer member function. Higher order function called from main.py"""

        # debug
        self.loss_track = {"step": [], "total_loss": [], "cont_loss": [], "styl_loss": [],
                           "cont_loss_list": [], "styl_loss_list": []}
        def helper(styl_loss, cont_loss, total_loss,
                   styl_loss_list, cont_loss_list, gen_cont_act, gen_styl_act,
                   styl_act, cont_act, image):

            # add loss to loss track
            _round_num = lambda num: format(num, '.2f')
            _round_arr = lambda l: [format(x, '.2f') for x in l]
            self.loss_track["step"].append(self.step)
            self.loss_track["total_loss"].append(_round_num(total_loss))
            self.loss_track["styl_loss"].append(_round_num(styl_loss))
            self.loss_track["cont_loss"].append(_round_num(cont_loss))
            self.loss_track["cont_loss_list"].append(_round_arr(cont_loss_list))
            self.loss_track["styl_loss_list"].append(_round_arr(styl_loss_list))

            # print loss
            print("Step: {}\tStyle Loss: {}\tContent Loss: {}\tTotal Loss: {}".format(self.step, styl_loss, cont_loss, total_loss))
            print("styl_loss_list: {}".format(styl_loss_list))
            print("cont_loss_list: {}".format(cont_loss_list))
            #print("gen_cont_act: {}".format(gen_cont_act[self.cont_layers[0]]))
            #print("cont_act: {}".format(cont_act[self.cont_layers[0]]))
            #print("gen_styl_act")
            #[print(l, np.sum(gen_styl_act[l]), np.sum(gen_styl_act[l]*0.2)) for l in self.styl_layers]
            #[print(l, np.sum(styl_act[l]), np.sum(styl[l])*0.2) for l in styl_layers]
            #for l in self.styl_layers:
            #    print("Layer: {}".format(l))
            #    a = np.sum(gen_styl_act[l])
            #    b = np.sum(styl_act[l])
            #    print(a-b)
            #print()
            #for l in self.cont_layers:
            #    print("Layer: {}".format(l))
            #    print(cont_loss_list[0])
            self.step += 1
        return helper


    def _gram(self, features):
        """Computes Gram matrix. Outputs (channel, channel) matrix."""
        _, h, w, c = features.get_shape()
        features_t = tf.transpose(features, perm=(0, 3, 1, 2))
        matrix = tf.reshape(features_t, shape=[c.value, h.value*w.value])
        return tf.matmul(matrix, matrix, transpose_b=True) / tf.size(matrix, out_type=tf.float32)
