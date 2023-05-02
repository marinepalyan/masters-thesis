import argparse
import json
import os
from datetime import datetime
from typing import List, Callable

import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import itertools

from models import get_model, MODELS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TOTAL_NUM_OF_USERS = 15
HR_GRID = list(range(30, 230, 1))
CONFIG = {}

LABEL_DISTRIBUTIONS = {
    'gaussian': tfp.distributions.Normal,
    'cauchy': tfp.distributions.Cauchy,
}
MODEL_CONFIG = {
    'regression': {
        'num_of_classes': 1,
        'output_activation': 'relu',
        'loss': 'mse',
    },
    'classification': {
        'num_of_classes': len(HR_GRID),
        'output_activation': 'softmax',
        'loss': 'categorical_crossentropy',
    }
}


def set_config(config_file):
    global CONFIG
    with open(config_file) as f:
        CONFIG = json.load(f)


def apply_to_keys(keys: List[str], func: Callable):
    def apply_to_keys_fn(example, training: bool):
        for key in keys:
            example[key] = func(example[key])
        return example
    return apply_to_keys_fn


def fill_zeros(features, label, training: bool = True):
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
        return tf.concat([tf.zeros([n]), tf.slice(column, [n], [CONFIG['sample_size'] - n])], axis=0)

    features = apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg'],
                             func=slice_zeros_wrapper)(features, training)
    return features, label


def choose_label(features, label, training: bool):
    if CONFIG['label'] == 'last':
        idx = -1
    elif CONFIG['label'] == 'middle':
        idx = CONFIG['sample_size'] // 2
    elif isinstance(CONFIG['label'], int):
        # If the label is a percentage, then choose the label at that percentile
        idx = int(CONFIG['sample_size'] * int(CONFIG['label']) / 100)
    return features, label['heart_rate'][idx]


def int_label(features, label, training: bool):
    return features, tf.math.round(label)


def one_hot_label(features, label, training: bool):
    label = tf.math.round(label)
    label = tf.cast(label, tf.int32)
    return features, tf.one_hot(label - HR_GRID[0], len(HR_GRID))


def dist_label(features, label, training: bool):
    label = tf.math.round(label)
    try:
        dist_function = LABEL_DISTRIBUTIONS[CONFIG['distribution']]
    except KeyError:
        raise ValueError(f"Invalid distribution {CONFIG['distribution']}. "
                         f"Must be one of {list(LABEL_DISTRIBUTIONS.keys())}")
    distribution = dist_function(loc=label, scale=CONFIG['scale'])
    return features, distribution.prob(HR_GRID)


def join_features(features, label, training: bool):
    features = tf.concat([features['acc_x'], features['acc_y'], features['acc_z'], features['ppg']], axis=0)
    features = tf.reshape(features, (CONFIG['sample_size'], 4))
    # label = tf.reshape(label, (1,))
    return features, label


# Define a function to parse each example in the TFRecord file
def parse_example(example_proto, training: bool):
    feature_description = {
        "acc_x": tf.io.VarLenFeature(tf.float32),
        "acc_y": tf.io.VarLenFeature(tf.float32),
        "acc_z": tf.io.VarLenFeature(tf.float32),
        "ppg": tf.io.VarLenFeature(tf.float32),
        "activity": tf.io.VarLenFeature(tf.int64),
        "heart_rate": tf.io.VarLenFeature(tf.float32)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example


def sample_dataset(example, training: bool):
    start_idx = tf.random.uniform((), minval=0, maxval=tf.shape(example['acc_x'])[0] - CONFIG['sample_size'] - 1, dtype=tf.int32)

    # tf.print(f"n: {n}")
    def slice_column(column):
        return column[start_idx:start_idx + CONFIG['sample_size']]

    keys = ['acc_x', 'acc_y', 'acc_z', 'ppg', 'heart_rate']
    example = (apply_to_keys(keys=keys, func=slice_column))(example, training)
    return example


def separate_features_label(dataset, training: bool):
    features = dataset.copy()
    label = {"heart_rate": features.pop('heart_rate')}
    return features, label


def build_dataset(ds, transforms, training=False):
    ds = ds.cache().repeat()
    for transform in transforms[0]:
        ds = ds.map(lambda x: transform(x, training=training))
    for transform in transforms[1]:
        ds = ds.map(lambda x, y: transform(x, y, training=training))
    # FIXME this is not working
    # getting an error that the dataset is not iterable
    # slice index -1 of dimension 0 out of bounds.
    for features, label in ds.take(5):
        print(features)
        print(features.shape)
        assert features.shape == (CONFIG['sample_size'], 4)
        print(label)
        print(label.shape)
    return ds


def prepare_data(input_files: List, test_size: float):
    test_users_size = int(TOTAL_NUM_OF_USERS * test_size)
    train_dataset = tf.data.TFRecordDataset(input_files[:-test_users_size])
    test_dataset = tf.data.TFRecordDataset(input_files[-test_users_size:])
    transforms = [
        [
            parse_example,
            apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg', 'activity', 'heart_rate'], func=tf.sparse.to_dense),
            sample_dataset,
            separate_features_label],
        [
            fill_zeros,
            choose_label,
            join_features
        ]
    ]
    if CONFIG['model_type'] == 'classification':
        if CONFIG['distribution'] is None:
            raise ValueError("Must specify a distribution for classification")
        if CONFIG['distribution'] == 'one_hot':
            transforms[1].append(one_hot_label)
        else:
            transforms[1].append(dist_label)
    train_ds = build_dataset(train_dataset, transforms, training=True)
    test_ds = build_dataset(test_dataset, transforms, training=False)

    train_ds = train_ds.shuffle(100).batch(32)
    test_ds = test_ds.batch(CONFIG['batch_size'])
    return train_ds, test_ds


# Define the main function
def main(input_files: List, test_size: float):
    train_ds, test_ds = prepare_data(input_files, test_size)
    main_work_dir = os.path.join("../logs", CONFIG['model_name'], CONFIG['model_type'])
    if CONFIG['distribution'] is not None:
        main_work_dir = os.path.join(main_work_dir, CONFIG['distribution'])
    os.makedirs(main_work_dir, exist_ok=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=main_work_dir)
    input_shape = (CONFIG['sample_size'], 4)
    # Create the TensorFlow model and compile it
    model = get_model(CONFIG['model_name'], MODEL_CONFIG[CONFIG["model_type"]], input_shape)
    # Train the model on the transformed dataset
    model.fit(train_ds, steps_per_epoch=CONFIG['steps_per_epoch'], epochs=CONFIG['epochs'],
              validation_data=test_ds, validation_steps=CONFIG['validation_steps'],
              callbacks=[tensorboard_callback])
    save_path = os.path.join(main_work_dir,
                             f"{CONFIG['model_name']}_{CONFIG['model_type']}_{CONFIG['distribution']}.ckpt")
    # model.save_weights(save_path)
    # save training config in the same folder
    with open(os.path.join(main_work_dir, 'config.json'), 'w') as f:
        json.dump(CONFIG, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--config_path', type=str, default='../basic_config.json')
    args = parser.parse_args()
    input_files = [f'../data/processed/S{user}.tfrecord' for user in range(1, TOTAL_NUM_OF_USERS + 1)]
    set_config(args.config_path)
    models_config = {
        'model_type': ['regression', 'classification'],
        'model_name': MODELS.keys(),
        'distribution': [None, 'one_hot', 'gaussian', 'cauchy'],
        'label': ['last', 'middle']
    }
    keys, values = zip(*models_config.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    permutations_dicts = [d for d in permutations_dicts if not (d['model_type'] == 'regression'
                                                                and d['distribution'] is not None)]
    permutations_dicts = [d for d in permutations_dicts if not (d['model_type'] == 'classification'
                                                                and d['distribution'] is None)]

    for permutation in permutations_dicts:
        CONFIG.update(permutation)
        CONFIG["sample_size"] = 1565 if CONFIG['model_name'] == 'tcn' else 1500
        print(CONFIG)
        main(input_files, args.test_size)
