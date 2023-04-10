from typing import Tuple

import tensorflow as tf

from .dense import get_dense_model
from .tcn import get_tcn_model

MODELS = {
    'dense': get_dense_model,
    'tcn': get_tcn_model,
}


def get_model(model_name: str, input_shape: Tuple[int, int],
              optimizer: str = 'adam', loss: str = 'mse') -> tf.keras.Model:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Must be one of {list(MODELS.keys())}")
    model = MODELS[model_name](input_shape)
    model.compile(optimizer=optimizer, loss=loss)
    print(model.summary())
    return model
