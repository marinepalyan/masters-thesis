import argparse
import json
import os
from datetime import datetime
from typing import List, Callable

import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SAMPLE_SIZE = 1500
input_shape = (SAMPLE_SIZE, 4)
TOTAL_NUM_OF_USERS = 15
HR_GRID = list(range(30, 230, 1))

CONFIG = {}

LOGDIR = '../logs/{}_{}.txt'


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


# specify to do this only when training
# how much future is useful? e.g. 1-minute data into the future
# after a while start losing past data and results get worse
# use not the last label, but in the middle, or 75th percentile
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
        return tf.concat([tf.zeros([n]), tf.slice(column, [n], [SAMPLE_SIZE - n])], axis=0)

    features = apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg'],
                             func=slice_zeros_wrapper)(features, training)
    return features, label


def choose_label(features, label, training: bool):
    return features, label['heart_rate'][-1]


def int_label(features, label, training: bool):
    return features, tf.cast(label, tf.int32)


def one_hot_label(features, label, training: bool):
    label = tf.cast(label, tf.int32)
    return features, tf.one_hot(label - HR_GRID[0], len(HR_GRID))


def normal_dist_label(features, label, training: bool):
    label = tf.cast(label, tf.int32)
    normal_distribution = tfp.distributions.Normal(loc=label, scale=CONFIG['scale'])
    return features, normal_distribution.prob(HR_GRID)


def cauchy_dist_label(features, label, training: bool):
    label = tf.cast(label, tf.int32)
    cauchy_distribution = tfp.distributions.Cauchy(loc=label, scale=CONFIG['scale'])
    return features, cauchy_distribution.prob(HR_GRID)


def join_features(features, label, training: bool):
    features = tf.concat([features['acc_x'], features['acc_y'], features['acc_z'], features['ppg']], axis=0)
    features = tf.reshape(features, input_shape)
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
    # TODO fix maxval
    start_idx = tf.random.uniform((), minval=0, maxval=tf.shape(example['acc_x'])[0] - SAMPLE_SIZE - 1, dtype=tf.int32)

    # tf.print(f"n: {n}")
    def slice_column(column):
        return column[start_idx:start_idx + SAMPLE_SIZE]

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
        print(label)
    return ds


# Define the main function
def main(input_files: List, test_size: float):
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=LOGDIR.format(CONFIG['model_name'], datetime.now().strftime("%Y%m%d-%H%M%S")))
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
    train_ds = build_dataset(train_dataset, transforms, training=True)
    test_ds = build_dataset(test_dataset, transforms, training=False)

    train_ds = train_ds.shuffle(100).batch(32)
    test_ds = test_ds.batch(CONFIG['batch_size'])

    # Create the TensorFlow model and compile it
    model = get_model(CONFIG['model_name'], input_shape)
    # Train the model on the transformed dataset
    model.fit(train_ds, steps_per_epoch=CONFIG['steps_per_epoch'], epochs=CONFIG['epochs'],
              validation_data=test_ds, validation_steps=CONFIG['validation_steps'],
              callbacks=[tensorboard_callback])

    save_path = f"{CONFIG['model_name']}_{CONFIG['model_type']}.ckpt"
    model.save_weights(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--config_path', type=str, default='../basic_config.json')
    args = parser.parse_args()
    input_files = [f'../data/processed/S{user}.tfrecord' for user in range(1, TOTAL_NUM_OF_USERS + 1)]
    set_config(args.config_path)
    for model in ['tcn']:
        CONFIG['model_name'] = model
        main(input_files, args.test_size)
