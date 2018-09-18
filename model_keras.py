"""
Date: Aug. 28, 2018
Author: Eric Cotner, UCLA Dept. of Physics and Astronomy

Constructs the neural network architecture, and specifies the training, prediction, saving/loading procedures.

xTODO: add support for learning rate function
xTODO: add support for tunable ReLU activation
xTODO: add support for learnable PReLU activation
xTODO: add method for saving NN weights
xTODO: add method for loading saved NN weights
TODO: add prediction method
TODO: add tensorboard monitoring
TODO: figure out way to save learning curve/progress to file
"""

import tensorflow as tf
from tensorflow import keras
K = keras.backend
KL = keras.layers
KM = keras.models
KE = keras.estimator
KU = keras.utils
KD = keras.datasets
import utils
import os

def tunable_relu(x, theta=0):
    """ Leaky relu with tunable slope given by tan(theta). """
    x = K.maximum(tf.tan(theta)*x, x)
    return x

class ThetaCallback(keras.callbacks.Callback):
    """ Callback for updating theta parameter based on schedule specified in config file. """
    def __init__(self, theta, config):
        super(ThetaCallback, self).__init__()
        self.theta = theta
        self.config = config

    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.theta, self.config.THETA_SCHEDULE(epoch))
        print(K.get_value(self.theta))

class Model(object):
    """
    Neural network model for convexity annealing experiment. Composed of several 2D convolutions.
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
        if self.config.THETA_TRAINABLE:     # TODO: build model using trainable prelu activations
            for L in self.config.ARCHITECTURE:
                if L[0] == "conv2d":
                    X = KL.Conv2D(**L[1])(X)
                    X = KL.PReLU(alpha_initializer='ones', shared_axes=[1,2,3])(X)
        else:                               # TODO: alter layers to allow for tunable theta
            self.theta = tf.Variable(initial_value=0.0, trainable=False, name="theta", dtype=tf.float32)
            for L in self.config.ARCHITECTURE:
                if L[0] == "conv2d":
                    X = KL.Conv2D(**L[1])(X)
                    X = KL.Lambda(tunable_relu, arguments={"theta": self.theta})(X)

        # Output layer
        X = KL.Flatten()(X)
        outputs = KL.Dense(self.config.OUTPUT_SHAPE, activation=self.config.OUTPUT_ACTIVATION)(X)

        # Create model, specify loss function, optimizer and metrics
        self.model = KM.Model(inputs=inputs, outputs=outputs)

    def train(self, data, labels, validation_data=None):
        """ Train the neural network. """

        # Specify optimizer, learning rate schedule, etc and compile model
        opt = keras.optimizers.Adam()
        callbacks = [keras.callbacks.LearningRateScheduler(lambda epoch: self.config.LR_FUNC(epoch), verbose=0)]
        if not self.config.THETA_TRAINABLE: callbacks.append(ThetaCallback(self.theta, self.config))
        self.model.compile(optimizer=opt,
                           loss=self.config.LOSS,
                           metrics=self.config.METRICS)

        # Begin training
        self.model.fit(data, labels, epochs=self.config.MAX_EPOCHS,
                       batch_size=self.config.BATCH_SIZE, validation_data=validation_data, shuffle=False,
                       callbacks=callbacks)

    def save(self):
        # Check to make sure directory exists, and make it if not
        try:
            os.mkdir(self.config.SAVE_PATH)
        except FileExistsError: pass
        # Only saves the weights, not architecture, since we can reconstruct the architecture by calling build_model()
        self.model.save_weights(self.config.SAVE_PATH + "KerasModel.h5")

    def load(self):
        # First build the model, then load weights
        self.build_model()
        self.model.load_weights(self.config.SAVE_PATH + "KerasModel.h5")

    def predict(self):
        pass