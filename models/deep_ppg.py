"""
The following model is an implemetation of the architecture described in the paper:
Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks
Reference: https://www.mdpi.com/1424-8220/19/14/3079
"""

from typing import Tuple

import tensorflow as tf


def get_deep_ppg_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    # Assuming input is 1025x4
    # conv to make it into 1025x8
    x = tf.keras.layers.Conv1D(8, kernel_size=1, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(16, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(256, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(512, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(1024, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model