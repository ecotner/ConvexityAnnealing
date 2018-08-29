"""
Date: Aug. 28, 2018
Author: Eric Cotner, UCLA Dept. of Physics and Astronomy

Specifies the neural network model and its training
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
K = keras.backend
KL = keras.layers
KM = keras.models
KE = keras.estimator
KU = keras.utils
KD = keras.datasets
#from config import Config
import utils

class Model(object):
    """
    Neural network model for convexity annealing experiment
    """

    def __init__(self, config):
        """
        Initializes the model for the experiment using the provided configuration.

        :param config: Config class containing all the parameters of the experiment
        """
        self.config = config
        self.model = None

    def build_model(self):
        """
        Constructs the model architecture
        :return:
        """

        # Set seeds
        utils.set_global_seed(self.config.SEED, use_parallelism=self.config.USE_PARALLELISM)

        # Input layer
        inputs = KL.Input(shape=self.config.INPUT_SHAPE)
        X = inputs

        # Hidden layers
        for L in self.config.ARCHITECTURE:
            if L[0] == "conv2d":
                X = KL.Conv2D(**L[1])(X)

        # Output layer
        X = KL.Flatten()(X)
        outputs = KL.Dense(self.config.OUTPUT_SHAPE, activation=self.config.OUTPUT_ACTIVATION)(X)

        # Create model, specify loss function, optimizer and metrics
        self.model = KM.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.config.OPTIMIZER,
                           loss=self.config.LOSS,
                           metrics=self.config.METRICS)

    def train(self, data, labels, validation_data=None):
        #assert data.shape[1:] == self.config.INPUT_SHAPE, "Input data not the right shape"
        self.model.fit(data, labels, epochs=self.config.MAX_EPOCHS,
                       batch_size=self.config.BATCH_SIZE, validation_data=validation_data, shuffle=False)

    def save(self):
        pass

    def load(self):
        pass

    def predict(self):
        pass