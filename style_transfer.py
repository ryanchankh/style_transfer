"""
Style Transfer Object
Last Updated: 17 June 2018
Author: Ryan Chan
"""

import tensorflow as tf
import numpy as np 
from PIL import Image
import time
from datetime import datetime

import vgg19.vgg as vgg
import utils



class StyleTransfer(): 
    """
    Style Transfer Object. 
    """
    
    def __init__(self, gen_img, img_shape, cont_layers, styl_layers, styl_layer_weights, styl_loss_alpha, cont_loss_beta, learning_rate):
        """
        Initialize object. 

        Parameters: 
            gen_img: the initialize image; image to be turned into 
            img_shape: the shape of the image
            cont_layers: the layers chosen to compute the content layers
            styl_layesr: the layers chosen to comupte the style layers
            styl_layer_weighs: the assicoiated weights for each style feature layer when computing style loss
            styl_loss_alpha: a scalar to adjust overall style loss
            cont_loss_beta: a sclar to adjust overall content loss

        """
        self.gen_img = gen_img
        self.img_shape = img_shape
        self.gen_layers = cont_layers + styl_layers
        self.cont_layers = cont_layers
        self.styl_layers = styl_layers
        self.styl_layer_weights = styl_layer_weights
        self.styl_loss_alpha = styl_loss_alpha
        self.cont_loss_beta = cont_loss_beta
        self.learning_rate = learning_rate
        self.graph = self.build_graph()
        
    def build_graph(self):
        """
        The graph for the whole style tranfser process. 
        """
        with tf.Graph().as_default() as g:
            with tf.variable_scope("image_input") as scope:
                self.image = tf.Variable(self.gen_img, trainable=True, dtype=tf.float32) 
                self.styl_img = tf.placeholder(tf.float32, shape=[1, None, None, None], name="style_image")
                self.cont_img = tf.placeholder(tf.float32, shape=[1, None, None, None], name="content_image")

            with tf.name_scope("vgg_model") as scope:
                self.vgg_net = vgg.Vgg19()

            with tf.name_scope("gen_activity") as scope:
                gen_model = self.vgg_net.build(self.image)
                self.gen_layer_ops = gen_model.layer_dict()
                self.gen_cont_activity = {layer_name: self.gen_layer_ops[layer_name] for layer_name in self.cont_layers}
                self.gen_styl_activity = {layer_name: self.gen_layer_ops[layer_name] for layer_name in self.styl_layers}

            with tf.name_scope("styl_activity") as scope:
                styl_model = self.vgg_net.build(self.styl_img)
                self.styl_layer_ops = styl_model.layer_dict(self.styl_layers)
                self.styl_activity_calc = {layer_name: self.styl_layer_ops[layer_name] for layer_name in self.styl_layers}
        
                self.styl_activity = {layer_name: tf.placeholder(tf.float32, shape=[1, None, None, None], name="styl_activity") for layer_name in self.styl_layers}
                
            with tf.name_scope("cont_activity") as scope:
                cont_model = self.vgg_net.build(self.cont_img)
                self.cont_layer_ops = cont_model.layer_dict(self.cont_layers)
                self.cont_activity_calc = {layer_name: self.cont_layer_ops[layer_name] for layer_name in self.cont_layers}
                self.cont_activity = {layer_name: tf.placeholder(tf.float32, shape=[1, None, None, None]) for layer_name in self.cont_layers}
                
            with tf.variable_scope("styl_loss") as scope:
                self.styl_loss = self.styl_loss()
                
            with tf.variable_scope("cont_loss") as scope:
                self.cont_loss = self.cont_loss()
                
            with tf.variable_scope("total_loss") as scope:
                self.total_loss = self.styl_loss_alpha * self.styl_loss + self.cont_loss_beta * self.cont_loss

            with tf.name_scope("train") as scope:
                self.beta1 = tf.Variable(0.9)
                self.beta2 = tf.Variable(0.999)
                self.episilon = tf.Variable(1)
                self.train_ops = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,var_list=self.image)
                
            with tf.name_scope("init") as scope:
                self.init_op = tf.global_variables_initializer()
            
        return g

    def grammian(self, features):
        """
        Compute grammain for the given layer.
        
        Parameters: 
            features: a tensor of shape (1, Height, Width, Channels)

        Return:
            grammian of the matrix. 
        """
        features_shape = tf.shape(features)
        H = features_shape[1]
        W = features_shape[2]
        C = features_shape[3]

        matrix = tf.reshape(features, shape=[-1, C])
        gram = tf.matmul(matrix, matrix, transpose_a = True)
        return gram

    def cont_loss(self):
        """
        Computes content loss.
        """
        layer_cont_losses = {}
        
        # compute loss for each layer
        for layer in self.cont_layers: 
            layer_cont_losses[layer] = tf.losses.mean_squared_error(self.gen_cont_activity[layer],self.cont_activity[layer])
        cont_loss = tf.reduce_sum(list(layer_cont_losses.values()))
        
        return cont_loss
    
    def styl_loss(self):
        """
        Compute style loss.
        """
        layer_styl_losses = {}
        
        # compute grammian for each layer
        for layer_name in self.styl_layers:
            channels = self.gen_styl_activity[layer_name].get_shape().as_list()[3]
            feature_map_size = self.img_shape[1] * self.img_shape[2]
            denom = 2 * (channels ** 2) * (feature_map_size ** 2)
            layer_styl_losses[layer_name] = (1 / denom) * tf.losses.mean_squared_error(self.grammian(self.gen_styl_activity[layer_name]), self.grammian(self.styl_activity[layer_name]))
        
        total_styl_loss = tf.reduce_sum([self.styl_layer_weights[layer_name] * layer_styl_losses[layer_name] for layer_name in self.styl_layers])
        return total_styl_loss

    