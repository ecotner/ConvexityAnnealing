"""
Date: August 28, 2018
Author: Eric Cotner, UCLA Dept. of Physics and Astronomy

Collection of utility functions.
"""

import os
import random as rn
import numpy as np
import tensorflow as tf
from tensorflow import keras
K = keras.backend

def set_global_seed(seed, use_parallelism=True):
    """ Sets all possible seeds for reproducibility. """
    os.environ['PYTHONHASHSEED'] = str(seed)
    rn.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # Set tensorflow to use only a single thread
    # Allows for 100% reproducibility at the cost of not being able to multithread or use GPU
    if not use_parallelism:
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

def precision_macro(y_true, y_pred, n_classes=10):
    """ Calculates macro precision metric """
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    val = []
    for label in range(n_classes):
        y = K.equal(y_true, label)
        yhat = K.equal(y_pred, label)
        TP = K.sum(K.cast(K.equal(K.equal(yhat, True), K.equal(y, True)), dtype=tf.int32))
        FP = K.sum(K.cast(K.equal(K.equal(yhat, True), K.equal(y, False)), dtype=tf.int32))
        val.append(TP/(TP+FP))
    return tf.reduce_mean(val)

def recall_macro(y_true, y_pred, n_classes=10):
    """ Calculates macro precision metric """
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    val = []
    for label in range(n_classes):
        y = K.equal(y_true, label)
        yhat = K.equal(y_pred, label)
        TP = K.sum(K.cast(K.equal(K.equal(yhat, True), K.equal(y, True)), dtype=tf.int32))
        FN = K.sum(K.cast(K.equal(K.equal(yhat, False), K.equal(y, True)), dtype=tf.int32))
        val.append(TP/(TP+FN))
    return tf.reduce_mean(val)

def fbeta_macro(y_true, y_pred, beta=1, n_classes=10):
    """ Calculates macro precision metric """
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    val = []
    for label in range(n_classes):
        y = K.equal(y_true, label)
        yhat = K.equal(y_pred, label)
        TP = K.sum(K.cast(K.equal(K.equal(yhat, True), K.equal(y, True)), dtype=tf.int32))
        FN = K.sum(K.cast(K.equal(K.equal(yhat, False), K.equal(y, True)), dtype=tf.int32))
        FP = K.sum(K.cast(K.equal(K.equal(yhat, True), K.equal(y, False)), dtype=tf.int32))
        val.append((1+beta**2)*TP/((1+beta**2)*TP+(beta**2)*FN+FP))
    return tf.reduce_mean(val)
