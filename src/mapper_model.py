import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    BatchNormalization,
    Activation,
    Conv1D,
    ZeroPadding1D,
    Reshape,
    Cropping1D,
    MaxPooling1D,
    Dropout,
    
)
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean


def mapper_model(input_shape=(192,), latent_dim=8):

    mapper_inputs = Input(shape=input_shape)
    
    x = Dense(128, activation="relu")(mapper_inputs)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(latent_dim, activation="relu")(x)
    
    return Model(mapper_inputs, x)