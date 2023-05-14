from typing import Tuple, Dict

import tensorflow as tf

from metrics.weighted_mae import WeightedMAE
from metrics.weighted_mse import WeightedMSE
from metrics.two_label_loss import my_loss_fn
from metrics.two_label_weighted_mae import TwoLabelWeightedMAE
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
              optimizer: str = 'adam') -> tf.keras.Model:
    MODEL_CONFIG = {
        'regression': {
            'num_of_classes': 1,
            'output_activation': 'relu',
            'loss': 'mse',
            'metrics': ['mae', 'mse']
        },
        'classification': {
            'num_of_classes': 200,
            'output_activation': 'softmax',
            'loss': 'categorical_crossentropy',
            'metrics': [WeightedMAE(), WeightedMSE()]
        },
        'personalization': {
            'num_of_classes': 200,
            'output_activation': 'softmax',
            'loss': my_loss_fn,
            'metrics': [TwoLabelWeightedMAE()]
        }
    }
    model_config: Dict = MODEL_CONFIG[model_type]
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Must be one of {list(MODELS.keys())}")
    kwargs = {
        'input_shape': input_shape,
        'num_of_classes': model_config['num_of_classes'],
        'output_activation': model_config['output_activation'],
    }
    model = MODELS[model_name](**kwargs)
    optimizer_f = tf.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer_f, loss=model_config['loss'], metrics=model_config['metrics'])
    print(model.summary())
    return model
