from typing import Tuple

import tensorflow as tf


def get_dense_model(input_shape: Tuple[int, int],
                    output_activation: str,
                    num_of_classes: int,
                    ) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_of_classes, activation=output_activation)
    ])
    return model
