"""
Test Functions for computing gram matrix.
Name: gram.py
Author: Ryan Chan (ryanchankh@berkeley.edu)
Date: 09 September 2018
"""


class Functions():

    def _gram(self, features):
        """Computes Gram matrix. Outputs (channel, channel) matrix."""
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
