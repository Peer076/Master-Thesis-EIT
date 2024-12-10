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
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(latent_dim, activation="relu")(x)
    
    # Convolutional layers
    #x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(mapper_inputs)
    #x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    #x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    #x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
    #x = Flatten()(x)
    #x = Conv1D(latent_dim, kernel_size=3, activation='relu', padding='same')(x)
   
    #x = BatchNormalization()(x)
    #x = MaxPooling1D(pool_size=2)(x)
    
    #x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    #x = MaxPooling1D(pool_size=2)(x)
    
    #x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    #x = MaxPooling1D(pool_size=2)(x)
    
    # Flatten and dense layers
    #x = Flatten()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = Dense(64, activation='relu')(x)
    #x = Dense(latent_dim, activation='relu')(x)
    

    return Model(mapper_inputs, x)