from typing import Tuple

import tensorflow as tf


def get_chatgpt_model(input_shape: Tuple[int, int],
                      output_activation: str,
                      num_of_classes: int,
                      ) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(256, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_of_classes, activation=output_activation)(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
