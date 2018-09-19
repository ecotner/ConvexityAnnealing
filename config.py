"""
Date: August 28, 2018
Author: Eric Cotner, UCLA Dept. of Physics and Astronomy

Base configuration class which contains all the useful information important to the configuration of the experiment.
"""

from math import pi as PI
import tensorflow as tf
from tensorflow import keras
K = keras.backend
import os

class Config(object):
    """
    Configuration class; contains all relevant information to the experiment configuration.
    """
    # Name of the experiment and seed
    NAME = "Baseline_InverseDecay"
    DESCRIPTION = "Baseline training with regular ReLU activations (i.e. the x<0 branch has slope zero)"
    SAVE_PATH = "./" + NAME + "/"
    SEED = 0

    # Specify hyperparameters
    MAX_EPOCHS = 100
    VAL_SPLIT = 0.05            # Fraction of training set to use for validation
    BATCH_SIZE = 32
    USE_PARALLELISM = True      # Whether to use parallel threads (set to False for 100% reproducibility (may not
                                # work with GPU))
    LEARNING_DECAY = "inverse"
    INITIAL_LR = 1.0e-3

    # Experiment-specific parameters (like angle theta of x<0 branch of prelu)
    THETA_TRAINABLE = False
    THETA_DECAY = "constant"
    INITIAL_THETA = 0.0

    # Shape of input
    INPUT_SHAPE = [28, 28, 1]     # MNIST input shape - 28x28 greyscale images

    # Shape of output
    OUTPUT_SHAPE = 10                   # 10 MNIST classes
    OUTPUT_ACTIVATION = "softmax"

    # Specify optimizer, loss to optimize, and metrics to track
    OPTIMIZER = "adam"
    LOSS = "categorical_crossentropy"
    METRICS = [keras.metrics.categorical_accuracy]


    # Specification of hidden layer architecture
    ARCHITECTURE = [("conv2d", {"filters":16, "kernel_size":(4,4), "strides":1, "padding":"valid", "activation":"linear"}),
                    ("conv2d", {"filters":16, "kernel_size":(3,3), "strides":1, "padding":"valid", "activation":"linear"}),
                    ("conv2d", {"filters":16, "kernel_size":(3,3), "strides":1, "padding":"valid", "activation":"linear"})]

    # Make experiment directory if it doesn't exist
    try:
        os.mkdir(SAVE_PATH)
    except FileExistsError: pass

    # Set learning decay
    if LEARNING_DECAY == "inverse":
        def LR_FUNC(self, epoch):   # Define learning rate schedule
            return self.INITIAL_LR/(epoch+1)
    elif LEARNING_DECAY == "linear":
        def LR_FUNC(self, epoch):   # Define learning rate schedule
            return self.INITIAL_LR * (1 - epoch/self.MAX_EPOCHS)
    else:
        raise Exception("Not valid learning decay type")

    # Set theta decay
    if THETA_DECAY == "constant":
        def THETA_SCHEDULE(self, epoch):
            return 0.0                                      # For baseline relu
    elif THETA_DECAY == "linear":
        def THETA_SCHEDULE(self, epoch):
            return (PI/4)*(1 - epoch/self.MAX_EPOCHS)      # For linear annealing of theta
    else:
        raise Exception("Not valid theta decay type")
