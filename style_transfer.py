import numpy as np
import tensorflow as tf

#from vgg19.vgg_keras import VGG19
from vgg19.vgg import VGG19


class StyleTransfer():

    def __init__(self, init_img, img_shape, cont_layers, styl_layers, styl_weights, alpha, beta, l_rate):
        self.init_img = init_img
        self.img_shape = img_shape
        self.cont_layers = cont_layers
        self.styl_layers = styl_layers
        self.styl_weights = styl_weights
        self.alpha = alpha
        self.beta = beta
        self.l_rate = l_rate

        self.graph = self.build_graph()
        self.step = 0

    def build_graph(self):
        with tf.Graph().as_default() as graph:
            with tf.variable_scope("inputs") as scope:
                self.image = tf.Variable(self.init_img, trainable=True, dtype=tf.float32)
                self.cont_img = tf.placeholder(tf.float32, shape=(1, None, None, 3), name="cont_img")
                self.styl_img = tf.placeholder(tf.float32, shape=(1, None, None, 3), name="styl_img")

            with tf.name_scope("activitiy") as scope:
                self.styl_model = VGG19().build(self.styl_img)
                self.styl_activity = self.styl_model.layer_dict(self.styl_layers)

                self.cont_model = VGG19().build(self.cont_img)
                self.cont_activity = self.cont_model.layer_dict(self.cont_layers)

                self.gen_model = VGG19().build(self.image)
                self.gen_styl_activity = self.gen_model.layer_dict(self.styl_layers)
                self.gen_cont_activity = self.gen_model.layer_dict(self.cont_layers)

            with tf.name_scope("losses") as scope:
                self.total_loss = self.styl_loss() + self.cont_loss()

            with tf.name_scope("train") as scope:
                self.train = tf.train.AdamOptimizer(self.l_rate).minimize(self.total_loss)
        return graph

    def cont_loss(self):
        with tf.name_scope("cont_loss") as scope:
            losses = []
            for layer in self.cont_layers:
#                losses.append(tf.losses.mean_squared_error(self.gen_cont_activity[layer], self.cont_activity[layer]))
                losses.append(tf.reduce_sum(tf.pow(self.gen_cont_activity[layer] - self.cont_activity[layer], 2)) / 2)
        return self.beta * tf.reduce_sum(losses)

    def styl_loss(self):
        with tf.name_scope("styl_loss") as scope:
            losses = {}
            for layer in self.styl_layers:
                img_shape = self.gen_styl_activity[layer].get_shape().as_list()
                channels = img_shape[3]
                feature_map_size = img_shape[1] * img_shape[2]
                styl_gram = self.grammian(self.styl_activity[layer])
                gen_gram = self.grammian(self.gen_styl_activity[layer])
                #losses[layer] = (1. / (2. * (channels ** 2) * (feature_map_size ** 2))) * tf.losses.mean_squared_error(styl_gram, gen_gram)
                print("losses: {}".format(losses))
                losses[layer] = (1. / (2. * (channels ** 2) * (feature_map_size ** 2))) * tf.reduce_sum(tf.pow(styl_gram - gen_gram, 2))

            return self.alpha * tf.reduce_sum([self.styl_weights[l] * losses[l] for l in self.styl_layers])

    def grammian(self, features):
        features_shape = tf.shape(features)
        matrix = tf.reshape(features, shape=[-1, features_shape[3]])
        return tf.matmul(matrix, matrix, transpose_a = True)

    def step_callback(self, logger, sess, save_per_step):
        def helper(image):
            if self.step % save_per_step == 0:
                image = np.reshape(image, newshape=self.img_shape)
                logger.save_img_step(self.step, image, self.img_shape)
            self.step += 1
        return helper

    def loss_callback(self):
        def helper(*args):
            print(*args)

        return helper
