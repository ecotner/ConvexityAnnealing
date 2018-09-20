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
        TP = tf.reduce_sum(K.cast(tf.logical_and(y, yhat), dtype=tf.float32))
        FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y), yhat), dtype=tf.float32))
        #TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y), tf.logical_not(yhat)), dtype=tf.float32))
        #FN = tf.reduce_sum(tf.cast(tf.logical_and(y, tf.logical_not(yhat)), dtype=tf.float32))
        val.append(tf.div(TP,TP+FP+1.0e-9))
    return tf.reduce_mean(val)

def recall_macro(y_true, y_pred, n_classes=10):
    """ Calculates macro recall metric """
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    val = []
    for label in range(n_classes):
        y = K.equal(y_true, label)
        yhat = K.equal(y_pred, label)
        TP = tf.reduce_sum(K.cast(tf.logical_and(y, yhat), dtype=tf.float32))
        #FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y), yhat), dtype=tf.float32))
        #TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y), tf.logical_not(yhat)), dtype=tf.float32))
        FN = tf.reduce_sum(tf.cast(tf.logical_and(y, tf.logical_not(yhat)), dtype=tf.float32))
        val.append(tf.div(TP, TP+FN+1.0e-9))
    return val[1]

def fbeta_macro(y_true, y_pred, beta=1, n_classes=10):
    """ Calculates macro precision metric """
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    val = []
    for label in range(n_classes):
        y = K.equal(y_true, label)
        yhat = K.equal(y_pred, label)
        TP = tf.reduce_sum(K.cast(tf.logical_and(y, yhat), dtype=tf.float32))
        FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y), yhat), dtype=tf.float32))
        #TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y), tf.logical_not(yhat)), dtype=tf.float32))
        FN = tf.reduce_sum(tf.cast(tf.logical_and(y, tf.logical_not(yhat)), dtype=tf.float32))
        val.append(tf.div((1+beta**2)*TP, (1+beta**2)*TP+(beta**2)*FN+FP + 1.0e-9))
    return tf.reduce_mean(val)

def test_metric_y_true(y_true, y_pred): print("y_true shape=", y_true); return K.shape(y_true)[1]
def test_metric_y_pred(y_true, y_pred): print("y_pred shape=", y_pred); return K.shape(y_pred)[1]
