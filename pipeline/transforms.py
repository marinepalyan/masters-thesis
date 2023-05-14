import os
from typing import List, Callable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from scipy.signal import butter, filtfilt

HR_GRID = list(range(30, 230, 1))

LABEL_DISTRIBUTIONS = {
    'gaussian': tfp.distributions.Normal,
    'cauchy': tfp.distributions.Cauchy,
}

__all__ = [
    'parse_example',
    'apply_to_keys',
    'sample_dataset',
    'separate_features_label',
    'apply_ppg_filter',
    'standardize_ppg',
    'fill_zeros',
    'choose_label',
    'int_label',
    'one_hot_label',
    'dist_label',
    'join_features',
    'double_sample_dataset',
    'predict_label',
    'expand_features_dims',
    'finalize_label'
]


def apply_to_keys(keys: List[str], func: Callable):
    def apply_to_keys_fn(example, training: bool, **CONFIG):
        for key in keys:
            example[key] = func(example[key])
        return example

    return apply_to_keys_fn


def fill_zeros(features, label, training: bool = True, **CONFIG):
    if not training:
        return features, label
    prob_to_run = np.random.randint(0, 4)
    # If the number is 0, then run the transform
    if prob_to_run == 0:
        return features, label
    # Generate a random number between 0 and 1250 inclusive
    n = tf.random.uniform([], maxval=1250, dtype=tf.int32)

    # Assign the first n values to zeros
    def slice_zeros_wrapper(column):
        return tf.concat([tf.zeros([n]), tf.slice(column, [n], [1565 - n])], axis=0)

    features = apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg'],
                             func=slice_zeros_wrapper)(features, training)
    return features, label


def choose_label(features, label, training: bool, **CONFIG):
    if CONFIG['choose_label'] == 'last':
        idx = -1
    elif CONFIG['choose_label'] == 'middle':
        idx = CONFIG['sample_size'] // 2
    elif isinstance(CONFIG['choose_label'], int):
        # If the label is a percentage, then choose the label at that percentile
        idx = int(CONFIG['sample_size'] * int(CONFIG['label']) / 100)
    # - 30
    return features, label['heart_rate'][idx]


def int_label(features, label, training: bool, **CONFIG):
    return features, tf.math.round(label)


def one_hot_label(features, label, training: bool, **CONFIG):
    label = tf.math.round(label)
    label = tf.cast(label, tf.int32)
    return features, tf.one_hot(label - HR_GRID[0], len(HR_GRID))


def dist_label(features, label, training: bool, **CONFIG):
    label = tf.math.round(label)
    try:
        dist_function = LABEL_DISTRIBUTIONS["gaussian"]
    except KeyError:
        raise ValueError(f"Invalid distribution {CONFIG['distribution']}. "
                         f"Must be one of {list(LABEL_DISTRIBUTIONS.keys())}")
    distribution = dist_function(loc=label, scale=5)
    return features, distribution.prob(HR_GRID)


def join_features(features, label, training: bool, **CONFIG):
    features = tf.concat([features['acc_x'], features['acc_y'], features['acc_z'], features['ppg']], axis=0)
    features = tf.reshape(features, (1565, 4))
    # label = tf.reshape(label, (1,))
    return features, label


# Define a function to parse each example in the TFRecord file
def parse_example(example_proto, training: bool, **CONFIG):
    feature_description = {
        "acc_x": tf.io.VarLenFeature(tf.float32),
        "acc_y": tf.io.VarLenFeature(tf.float32),
        "acc_z": tf.io.VarLenFeature(tf.float32),
        "ppg": tf.io.VarLenFeature(tf.float32),
        # "activity": tf.io.VarLenFeature(tf.int64),
        "heart_rate": tf.io.VarLenFeature(tf.float32)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example


def sample_dataset(example, training: bool, **CONFIG):
    start_idx = tf.random.uniform((), minval=0, maxval=tf.shape(example['acc_x'])[0] - 1565 - 1, dtype=tf.int32)

    # tf.print(f"n: {n}")
    def slice_column(column):
        return column[start_idx:start_idx + 1565]

    keys = ['acc_x', 'acc_y', 'acc_z', 'ppg', 'heart_rate']
    example = (apply_to_keys(keys=keys, func=slice_column))(example, training)
    return example


def separate_features_label(dataset, training: bool, **CONFIG):
    features = dataset.copy()
    label = {"heart_rate": features.pop('heart_rate')}
    return features, label


# noinspection PyTupleAssignmentBalance
def apply_ppg_filter(features, label, training: bool, **CONFIG):
    if CONFIG['use_ppg_filter']:
        b, a = butter(4, 0.5, 'highpass', fs=25, output='ba')
        features['ppg'] = tf.py_function(filtfilt, [b, a, features['ppg']], tf.float32)
    return features, label


def standardize_ppg(features, label, training: bool, **CONFIG):
    mean, variance = tf.nn.moments(features['ppg'], axes=[0])
    std = tf.sqrt(variance)
    features['ppg'] = (features['ppg'] - mean) / std
    return features, label


def double_sample_dataset(example, training: bool, **CONFIG):
    offset = CONFIG['sample_size'] // 2
    start_idx = tf.random.uniform((), minval=0,
                                  maxval=tf.shape(example['acc_x'])[0] - (CONFIG['sample_size'] + offset) - 1,
                                  dtype=tf.int32)

    def slice_column(column):
        return column[start_idx:start_idx + CONFIG['sample_size']]

    def slice_column_with_offset(column):
        return column[start_idx + offset:start_idx + offset + CONFIG['sample_size']]

    keys = ['acc_x', 'acc_y', 'acc_z', 'ppg', 'heart_rate']
    rt_example = (apply_to_keys(keys=keys, func=slice_column))(example.copy(), training)
    oracle_example = (apply_to_keys(keys=keys, func=slice_column_with_offset))(example.copy(), training)
    return rt_example, oracle_example


def predict_label(rt, oracle, training: bool, **CONFIG):
    feature, label = oracle
    oracle_prediction = CONFIG['oracle_model'](feature)
    return rt, oracle_prediction


def expand_features_dims(features, label, training: bool, **CONFIG):
    features = tf.expand_dims(features, axis=0)
    return features, label


def finalize_label(rt, oracle_pred, training: bool, **CONFIG):
    features, label = rt
    label = tf.reshape(label, (1, 200))
    # join label and oracle_pred
    new_label = tf.concat([label, oracle_pred], axis=0)
    return features, new_label
