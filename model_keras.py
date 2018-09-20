"""
Date: Aug. 28, 2018
Author: Eric Cotner, UCLA Dept. of Physics and Astronomy

Wrapper class for Keras model.
Constructs the neural network architecture, and specifies the training, prediction, saving/loading procedures.

"""

##############################################
################### IMPORTS ##################
##############################################

import tensorflow as tf
from tensorflow import keras
K = keras.backend
KL = keras.layers
KM = keras.models
KC = keras.callbacks
import utils
import os
import time

###############################################
############### KERAS CALLBACKS ###############
###############################################

def tunable_relu(x, theta=0):
    """ Leaky relu with tunable slope given by tan(theta). """
    x = K.maximum(tf.tan(theta)*x, x)
    return x

class ThetaCallback(KC.Callback):
    """ Callback for updating theta parameter based on schedule specified in config file. """
    def __init__(self, theta, config):
        super(ThetaCallback, self).__init__()
        self.theta = theta
        self.config = config

    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.theta, self.config.THETA_SCHEDULE(epoch))

class CSVLoggerCustom(KC.CSVLogger):
    def __init__(self, logdir, model, separator=",", append=False):
        super().__init__(filename=logdir+"log.csv", separator=separator, append=append)
        self.mod = model
        self.logdir = logdir
        self.start_time = None

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self.start_time = time.time()
        with open(self.logdir+"hyperparameters.log", "w+") as fo:
            fo.write(self.mod.config.NAME + " hyperparameters\n")
            fo.write("Date/time: {}\n\n".format(time.ctime(self.start_time)))
            for param in ["DESCRIPTION", "SEED", "MAX_EPOCHS", "VAL_SPLIT", "BATCH_SIZE",
                          "LEARNING_DECAY", "INITIAL_LR",
                          "REGULARIZER", "REGULARIZATION_COEFFICIENT",
                          "THETA_TRAINABLE", "THETA_DECAY", "INITIAL_THETA",
                          "USE_PARALLELISM", "OPTIMIZER", "LOSS", "ARCHITECTURE"]:
                fo.write("{}: {}\n".format(param, getattr(self.mod.config, param)))

    def on_epoch_end(self, epoch, logs=None):
        # Save theta and learning rate
        logs["learning_rate"] = self.mod.config.LR_FUNC(epoch)
        logs["theta"] = K.get_value(self.mod.theta)
        # Save time elapsed since previous epoch
        logs["train_time"] = time.time() - self.start_time
        # Write log to file
        super().on_epoch_end(epoch=epoch, logs=logs)

###################################################
################## KERAS MODEL ####################
###################################################


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
        self.theta = tf.Variable(initial_value=0.0, trainable=False, name="theta", dtype=tf.float32)

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

        # Set regularizer
        if self.config.REGULARIZER == "L1": reg_func = keras.regularizers.l1(self.config.REGULARIZATION_COEFFICIENT)
        elif self.config.REGULARIZER == "L2": reg_func = keras.regularizers.l2(self.config.REGULARIZATION_COEFFICIENT)
        else: raise Exception("Unknown regularizer")

        # Hidden layers
        for L in self.config.ARCHITECTURE:
            if L[0] == "conv2d":
                X = KL.Conv2D(**L[1], kernel_regularizer=reg_func)(X)
                if L[2]["pooling"] is not None:
                    X = KL.MaxPool2D(pool_size=(2,2))(X)
            elif L[0] == "dense":
                X = KL.Dense(**L[1], kernel_regularizer=reg_func)(X)

            # Activation functions
            if self.config.THETA_TRAINABLE:
                X = KL.PReLU(alpha_initializer='ones', shared_axes=[1,2,3])(X)
            else:
                X = KL.Lambda(tunable_relu, arguments={"theta": self.theta})(X)

        # Output layer
        X = KL.Flatten()(X)
        outputs = KL.Dense(self.config.OUTPUT_SHAPE, activation=self.config.OUTPUT_ACTIVATION)(X)

        # Create model, specify loss function, optimizer and metrics
        self.model = KM.Model(inputs=inputs, outputs=outputs)

        # Specify optimizer, learning rate schedule, etc and compile model
        opt = keras.optimizers.Adam()
        self.model.compile(optimizer=opt,
                           loss=self.config.LOSS,
                           metrics=self.config.METRICS)

    def train(self, data, labels, validation_data=None):
        """ Train the neural network. """
        # Specify callbacks
        callbacks = [KC.LearningRateScheduler(lambda epoch: self.config.LR_FUNC(epoch),
                                                           verbose=0),
                     KC.TensorBoard(log_dir=self.config.SAVE_PATH + "logs",
                                                 histogram_freq=1,
                                                 write_graph=False,
                                                 write_grads=False),
                     CSVLoggerCustom(logdir=self.config.SAVE_PATH+"logs/",
                                     model=self,
                                     append=False),
                     KC.ModelCheckpoint(filepath=self.config.SAVE_PATH+"ModelWeights.h5",
                                                     monitor="val_loss",
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1)]
        if not self.config.THETA_TRAINABLE: callbacks.append(ThetaCallback(self.theta, self.config))

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
        self.model.save_weights(self.config.SAVE_PATH + "ModelWeights.h5")

    def load(self):
        # First build the model, then load weights, then compile
        self.build_model()
        self.model.load_weights(self.config.SAVE_PATH + "ModelWeights.h5")
        opt = keras.optimizers.Adam()
        self.model.compile(optimizer=opt,
                           loss=self.config.LOSS,
                           metrics=self.config.METRICS)

    def predict(self, X):
        return self.model.predict(X)