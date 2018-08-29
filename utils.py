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
