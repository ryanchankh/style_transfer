import numpy as np
import tensorflow as tf


from vgg19.vgg_mat import VGG19

class StyleTransfer():

    def __init__(self, init_img, cont_img, styl_img, img_shape, cont_layers, styl_layers, styl_weights, alpha, beta):
        # images
        self.init_img = self.init_img(init_img, img_shape)
        self.cont_img = cont_img
        self.styl_img = styl_img

        # VGG graph
        self.vgg_net = VGG19()

        # image shape
        self.img_shape = img_shape

        # layers
        self.cont_layers = cont_layers
        self.styl_layers = styl_layers

        # weights
        self.styl_weights = styl_weights
        self.alpha = alpha
        self.beta = beta

        # graph
        self.graph = self.build_graph()
        self.step = 0

    def build_graph(self):
        with tf.Graph().as_default() as graph:
            with tf.variable_scope("inputs") as scope:
                self.img = tf.Variable(self.init_img, trainable=True, dtype=tf.float32)

            with tf.name_scope("activity") as scope:
                self.styl_act = self._activity(self.styl_img, self.styl_layers)
                self.cont_act= self._activity(self.cont_img, self.cont_layers)

                self.gen_styl_act = self._activity(self.img, self.styl_layers)
                self.gen_cont_act = self._activity(self.img, self.cont_layers)

            with tf.name_scope("losses") as scope:
                self.styl_loss = self._styl_loss()
                self.cont_loss = self._cont_loss()
                self.total_loss = self.alpha * self.styl_loss + self.beta * self.cont_loss

        return graph

    def step_callback(self, logger, sess, save_per_step):
        def helper(image):
            if self.step % save_per_step == 0:
                image = np.reshape(image, newshape=self.img_shape)
                logger.save_img_step(self.step, image, self.img_shape)
        return helper

    def loss_callback(self):
        def helper(styl_loss, cont_loss, total_loss):
            print("Step: {}\tStyle Loss: {}\tContent Loss: {}\tTotal Loss: {}".format(self.step, styl_loss, cont_loss, total_loss))
            self.step += 1
        return helper

    def _activity(self, img, layers):
        model = self.vgg_net.build(img)
        activites = model.layer_dict(layers)
        return activites

    def _cont_loss(self):
        with tf.name_scope("cont_loss") as scope:
            losses = []
            for layer in self.cont_layers:
                losses.append(tf.reduce_sum(tf.pow(self.gen_cont_act[layer] - self.cont_act[layer], 2)) / 2.)
        return self.beta * tf.reduce_sum(losses)

    def _styl_loss(self):
        with tf.name_scope("styl_loss") as scope:
            losses = 0.
            for l in self.styl_layers:
                _, height, width, channels = self.gen_styl_act[l].get_shape()
                feature_size = int(height * width)
                channels = int(channels)
                gen_styl_gram = self.grammian(self.gen_styl_act[l])
                styl_gram = self.grammian(self.styl_act[l])
                losses += self.styl_weights[l] * (1 / (4*feature_size*channels)) * tf.reduce_sum(tf.pow((gen_styl_gram - styl_gram), 2))
        return losses

    def grammian(self, features):
        features_shape = tf.shape(features)
        matrix = tf.reshape(features, shape=[-1, features_shape[3]])
        return tf.matmul(matrix, matrix, transpose_a = True)

    def init_img(cont_img, styl_img, img_shape, choice="rand_uni"):
        if choice == "cont":
            return cont_img.copy()
        elif choice == "styl":
            return styl_img.copy()
        elif choice == "rand_norm":
            return np.float32(np.random.normal(128, 50, size=img_shape))
        else:
            return np.float32(np.random.uniform(0, 255, size=img_shape))
