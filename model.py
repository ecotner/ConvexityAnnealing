"""
Date: July 21, 2018
Author: Eric Cotner, UCLA Dept. of Physics and Astronomy

Convenience class for simple CNN with trainable PReLU activations for use in convexity annealing experiments.
"""

import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.input = None
        self.labels = None
        self.output = None
        self.loss = None

    def build_model(self, alpha_initial_value, alpha_trainable):
        """ Constructs a simple CNN with trainable PReLU activations. """

        # Define utility functions
        def weight_variable(shape, name="weight"):
            W_initial = tf.get_variable(shape=shape, dtype=tf.float32, name=name,
                                        initializer=tf.contrib.layers.xavier_initializer())
            return tf.Variable(W_initial, name=name)

        def bias_variable(shape, name="bias"):
            b_initial = tf.constant(shape=shape, value=0.0)
            return tf.Variable(b_initial, name=name)

        def conv2d(X, W, s=1, name="conv"):
            conv = tf.nn.conv2d(input=X, filter=W, strides=[1, s, s, 1], padding="SAME", data_format="NHWC", name=name)
            return conv

        def prelu(X, alpha_initial_value, alpha_trainable=False, name="prelu", alpha_name="prelu_alpha"):
            X_shape = tf.shape(X)[1:]
            alpha = tf.Variable(initial_value=alpha_initial_value*tf.ones(shape=X_shape, dtype=tf.float32),
                                trainable=alpha_trainable, name=alpha_name)
            a = tf.maximum(alpha * X, X, name=name)
            return a

        with self.graph.as_default():
            with tf.name_scope("Inputs"):
                self.input = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="input")
                self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")

            with tf.name_scope("Layer1"):
                filter1 = weight_variable(shape=[4, 4, 1, 16], name="filter1")
                bias1 = bias_variable(shape=[16], name="bias1")
                conv1 = prelu(conv2d(self.input, filter1, s=2) + bias1, alpha_initial_value=alpha_initial_value,
                              alpha_trainable=alpha_trainable, name="conv1", alpha_name="alpha1")
            with tf.name_scope("Layer2"):
                filter2 = weight_variable(shape=[4, 4, 16, 16], name="filter2")
                bias2 = bias_variable(shape=[16], name="bias2")
                conv2 = prelu(conv2d(conv1, filter2, s=2) + bias2, alpha_initial_value=alpha_initial_value,
                              alpha_trainable=alpha_trainable, name="conv2", alpha_name="alpha2")
                flatten2 = tf.contrib.layers.flatten(conv2)

            with tf.name_scope("Output"):
                weight3 = weight_variable(shape=[7*7*16, 10], name="weight3")
                bias3 = bias_variable(shape=[10], name="bias3")
                logits = tf.matmul(flatten2, weight3) + bias3
                self.output = tf.nn.softmax(logits=logits, name="output")
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name="loss")




if __name__ == "__main__":
    model = Model()
    model.build_model(alpha_initial_value=1.0, alpha_trainable=False)
    print(model.graph.get_tensor_by_name("weight3:0"))