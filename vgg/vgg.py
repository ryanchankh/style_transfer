import tensorflow as tf
import numpy as np

class VGG19():
    """VGG19 Convolutional Neural Network.

    Note on Input Image:
        + Input image should be preprocessed by:
            1) Flip RGB to BGR channel orders
            2) subtracted by mean pixel value.
        + Input dimension should be [batch, height, width, 3].

    Weights' values range [0, 1].
    """
    def __init__(self, vgg19_npy_path="./vgg/vgg19.npy"):
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()

    def build(self, rgb):
        self.conv1_1 = self.conv_layer(rgb, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.avg_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.avg_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.avg_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.avg_pool(self.conv5_4, 'pool5')
        return self

    def layer_dict(self, layer_list=None):
        """A dictionary of the activations of each layer.

        @params:
            layer_list: if specified, return the layers in layer_list (Defult: None)

        @return:
            activations: {layer_name: activated layer}
        """

        result = {}
        result["conv1_1"] = self.conv1_1
        result["conv1_2"] = self.conv1_2
        result["poo1"] = self.pool1

        result["conv2_1"] = self.conv2_1
        result["conv2_2"] = self.conv2_2
        result["pool2"] = self.pool2

        result["conv3_1"] = self.conv3_1
        result["conv3_2"] = self.conv3_2
        result["conv3_3"] = self.conv3_3
        result["conv3_4"] = self.conv3_4
        result["pool3"] = self.pool3

        result["conv4_1"] = self.conv4_1
        result["conv4_2"] = self.conv4_2
        result["conv4_3"] = self.conv4_3
        result["conv4_4"] = self.conv4_4
        result["pool4"] = self.pool4

        result["conv5_1"] = self.conv5_1
        result["conv5_2"] = self.conv5_2
        result["conv5_3"] = self.conv5_3
        result["conv5_4"] = self.conv5_4
        result["pool5"] = self.pool5
        if layer_list is None:
            return result
        else:
            return {layer: result[layer] for layer in layer_list}

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")
