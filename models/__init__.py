from typing import Tuple

import tensorflow as tf

from .dense import DenseModel

MODELS = {
    'dense': DenseModel
}


def get_model(model_name: str, input_shape: Tuple[int, int],
              optimizer: str = 'adam', loss: str = 'mse') -> tf.keras.Model:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Must be one of {list(MODELS.keys())}")
    model = MODELS[model_name](input_shape)
    model.compile(optimizer=optimizer, loss=loss)
    model.build(input_shape)
    print(model.summary())
    return model
