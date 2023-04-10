from typing import Tuple

import tensorflow as tf


def get_dense_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    return model
