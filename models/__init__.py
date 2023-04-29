from typing import Tuple

import tensorflow as tf

from .dense import get_dense_model
from .tcn import get_tcn_model
from .chatgpt import get_chatgpt_model
from .deep_ppg import get_deep_ppg_model
from .wavenet import get_wavenet_model

MODELS = {
    'dense': get_dense_model,
    'tcn': get_tcn_model,
    'chatgpt': get_chatgpt_model,
    'deep_ppg': get_deep_ppg_model,
    'wavenet': get_wavenet_model,
}


def get_model(model_name: str, model_type: str, input_shape: Tuple[int, int],
              optimizer: str = 'adam', loss: str = 'mse') -> tf.keras.Model:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Must be one of {list(MODELS.keys())}")
    kwargs = {
        'input_shape': input_shape,
        'num_of_classes': 1 if model_type == 'regression' else 200,
        'output_activation': 'relu' if model_type == 'regression' else 'softmax'
    }
    model = MODELS[model_name](**kwargs)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])
    print(model.summary())
    return model
