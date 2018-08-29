"""
Date: August 28, 2018
Author: Eric Cotner, UCLA Dept. of Physics and Astronomy

Gathers MNIST data, splits it into training, validation and test sets, then initializes a model and trains on it.
"""

import os
import random as rn
import numpy as np
import tensorflow as tf
from tensorflow import keras
K = keras.backend
KU = keras.utils
from config import Config
c = Config()
from model_keras import Model
import utils

# Set seeds
utils.set_global_seed(c.SEED, use_parallelism=c.USE_PARALLELISM)

# Download the MNIST dataset
# X_train.shape = (60000, 28, 28)
# y_train.shape = (60000,) (the elements are the actual labels)
# X_test.shape = (10000, 28, 28)
# y_test.shape = (10000,)
MNIST = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = MNIST.load_data()
X_train = X_train[:5000]
y_train = y_train[:5000]

# Preprocess the data (reshape, rescale, etc.)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# Preprocess class labels
y_train = KU.to_categorical(y_train, num_classes=10)
y_test = KU.to_categorical(y_test, num_classes=10)

# Split into train/val sets
split_idx = int(c.VAL_SPLIT * X_train.shape[0])
X_val = X_train[:split_idx]
y_val = y_train[:split_idx]
X_train = X_train[split_idx:]
y_train = y_train[split_idx:]


# Create model from config
M = Model(c)
M.build_model()

# Initiate training
M.train(X_train, y_train, validation_data=(X_val, y_val))
"loss: 0.6031 - acc: 0.8206 - val_loss: 0.4208 - val_acc: 0.8920"