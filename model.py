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
        self.alpha = None

    def build_model(self, alpha_initial_value, alpha_trainable, seed=0):
        """ Constructs a simple CNN with trainable PReLU activations.
        float alpha_initial_value: the initial value for the slope of the x<0 regime of the PReLU
        bool alpha_trainable: whether or not the PReLU slope is trainable."""


        # Define alpha and seed
        tf.set_random_seed(seed)

        with self.graph.as_default():
            if alpha_trainable:
                self.alpha = tf.Variable(initial_value=float(alpha_initial_value), trainable=alpha_trainable, name="alpha")
            else:
                self.alpha = tf.placeholder_with_default(input=1.0, shape=[], name="alpha")

        # Define utility functions
        def weight_variable(shape, name="weight"):
            W_initial = tf.get_variable(shape=shape, dtype=tf.float32, name=name,
                                        initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            return tf.Variable(W_initial, name=name)

        def bias_variable(shape, name="bias"):
            b_initial = tf.constant(shape=shape, value=0.0)
            return tf.Variable(b_initial, name=name)

        def conv2d(X, W, s=1, name="conv"):
            conv = tf.nn.conv2d(input=X, filter=W, strides=[1, s, s, 1], padding="SAME", data_format="NHWC", name=name)
            return conv

        def prelu(X, name="prelu"):
            a = tf.maximum(self.alpha * X, X, name=name)
            return a

        with self.graph.as_default():
            with tf.name_scope("Inputs"):
                self.input = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="input")
                self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")

            with tf.name_scope("HiddenLayer1"):
                filter1 = weight_variable(shape=[4, 4, 1, 16], name="filter1")
                bias1 = bias_variable(shape=[16], name="bias1")
                conv1 = prelu(conv2d(self.input, filter1, s=2) + bias1, name="conv1")
            with tf.name_scope("HiddenLayer2"):
                filter2 = weight_variable(shape=[4, 4, 16, 16], name="filter2")
                bias2 = bias_variable(shape=[16], name="bias2")
                conv2 = prelu(conv2d(conv1, filter2, s=2) + bias2, name="conv2")
                flatten2 = tf.contrib.layers.flatten(conv2)

            with tf.name_scope("Output"):
                weight3 = weight_variable(shape=[7*7*16, 10], name="weight3")
                bias3 = bias_variable(shape=[10], name="bias3")
                logits = tf.add(tf.matmul(flatten2, weight3), bias3, name="logits")
                self.output = tf.nn.softmax(logits=logits, name="output")
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name="loss")

    def train(self, X, Y, max_epochs, batch_size, lr_func, conv_func):
        pass




if __name__ == "__main__":
    # Test that model compiles without errors, and seeds allow for reproducible training
    model = Model()
    model.build_model(alpha_initial_value=1.0, alpha_trainable=False, seed=0)
    np.random.seed(0)
    X = np.random.randn(1, 28, 28, 1)
    print("mean(X) = ", np.mean(X))
    print(model.graph.get_tensor_by_name("weight3:0"))
    print(model.alpha)
    with model.graph.as_default():
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            result = sess.run(model.output, feed_dict={model.input: X}).squeeze()
            ground_truth = np.array([0.061081048, 0.059048973, 0.10069974, 0.10454267, 0.05780006, 0.07840316,
                                      0.22537023, 0.16293015, 0.05672105, 0.093402885], dtype=np.float32)
            consistency_check = np.all(result == ground_truth)
            print("Consistency check passed: ", consistency_check)
            if not consistency_check:
                print(result)
                print(ground_truth)
