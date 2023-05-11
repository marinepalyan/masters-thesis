import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pprint import pprint
from typing import List

import tensorflow as tf
from matplotlib import pyplot as plt

import numpy as np

from models import get_model
from pipeline.training_pipeline import (apply_to_keys, separate_features_label, apply_ppg_filter,
                                        standardize_ppg, join_features, choose_label, one_hot_label)


CONFIG = {
    "sample_size": 1565,
    "device": "/gpu:0",
}

oracle_model_path = '/home/marine/learning/masters-thesis/logs/logs/20230508/tcn/classification/middle/one_hot/ppg_filter/'
# TODO deal with different input sizes
# rt_model_path = '/home/marine/learning/masters-thesis/logs/logs/20230509/wavenet/classification/last/one_hot/ppg_filter/'
rt_model_path = '/home/marine/learning/masters-thesis/logs/logs/20230508/tcn/classification/last/one_hot/ppg_filter/'


one_device_strategy = tf.distribute.OneDeviceStrategy(device=CONFIG['device'])
print(tf.config.list_physical_devices())
with one_device_strategy.scope():
    oracle_model = get_model('tcn', 'classification', (CONFIG['sample_size'], 4))
    latest = tf.train.latest_checkpoint(oracle_model_path)
    oracle_model.load_weights(latest).expect_partial()


FEATURE_TYPE_MAP = {
    'sensor_readings.105.timestamp (ms)': tf.float32,
    'sensor_readings.105.slot1.pd1': tf.float32,
    'sensor_readings.105.slot1.pd2': tf.float32,
    'sensor_readings.105.slot1.pd3': tf.float32,
    'sensor_readings.105.accelerometer_reading.x': tf.float32,
    'sensor_readings.105.accelerometer_reading.y': tf.float32,
    'sensor_readings.105.accelerometer_reading.z': tf.float32,
    'sensor_readings.200.timestamp (ms)': tf.float32,
    'sensor_readings.200.value': tf.float32,
}
feature_description = {k: tf.io.VarLenFeature(FEATURE_TYPE_MAP[k]) for k in FEATURE_TYPE_MAP.keys()}


def format_input(example):
    heart_rate = example['sensor_readings.200.value']
    acc_x = example['sensor_readings.105.accelerometer_reading.x'] / 250
    acc_y = example['sensor_readings.105.accelerometer_reading.y'] / 250
    acc_z = example['sensor_readings.105.accelerometer_reading.z'] / 250
    ppg = (example['sensor_readings.105.slot1.pd1'] +
           example['sensor_readings.105.slot1.pd2'] +
           example['sensor_readings.105.slot1.pd3']) / 3
    timestamp = example['sensor_readings.105.timestamp (ms)']
    return {
        'heart_rate': heart_rate,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'ppg': - ppg,
        # 'timestamp': timestamp
    }


def sample_dataset(example, training: bool):
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


def predict_label(rt, oracle, training: bool):
    feature, label = oracle
    oracle_prediction = oracle_model(feature)
    return rt, oracle_prediction


def expand_features_dims(features, label, training: bool):
    features = tf.expand_dims(features, axis=0)
    return features, label


def finalize_label(rt, oracle_pred, training: bool):
    features, label = rt
    # join label and oracle_pred
    new_label = tf.concat([label, oracle_pred], axis=0)
    return features, new_label


def create_and_parse_dataset(input_files: List):
    raw_dataset = tf.data.TFRecordDataset(input_files)

    parsed_dataset = raw_dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
    parsed_dataset = parsed_dataset.map(
        lambda x: apply_to_keys(keys=feature_description.keys(), func=tf.sparse.to_dense)(x, True))
    parsed_dataset = parsed_dataset.map(format_input)
    lengths = []
    # for elem in parsed_dataset.as_numpy_iterator():
    #     lengths.append(len(elem['acc_x']))
    #     # break

    # print(sum(lengths) / (25 * 3600)) # 16.733533333333334 hours (44 minutes)
    parsed_dataset = parsed_dataset.cache().repeat()
    parsed_dataset = parsed_dataset.map(lambda x: sample_dataset(x, True))
    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (separate_features_label(rt, True),
                                                            separate_features_label(oracle, True)))

    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (apply_ppg_filter(*rt, True),
                                                            apply_ppg_filter(*oracle, True)))
    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (standardize_ppg(*rt, True),
                                                            standardize_ppg(*oracle, True)))
    # for rt, oracle in parsed_dataset.take(5):
    #     from matplotlib import pyplot as plt
    #     plt.plot(rt[0]['ppg'])
    #     plt.plot(oracle[0]['ppg'])
    #     plt.legend(['rt', 'oracle'])
    #     plt.show()
    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (rt, join_features(*oracle, True)))
    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (rt, expand_features_dims(*oracle, True)))

    for ex1, ex2 in parsed_dataset.take(1):
        pprint(ex1)
        pprint(ex2)

    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (predict_label(rt, oracle, True)))
    # for ex1, ex2 in parsed_dataset.take(5):
    #     print(ex2.numpy().squeeze())
    #     print(ex1[1].numpy().squeeze())
    parsed_dataset = parsed_dataset.map(lambda rt, oracle_pred: (choose_label(*rt, True), oracle_pred))
    for ex1, ex2 in parsed_dataset.take(5):
        print(ex2.numpy().squeeze())
        print(ex1[1].numpy().squeeze())
    parsed_dataset = parsed_dataset.map(lambda rt, oracle_pred: (one_hot_label(*rt, True), oracle_pred))

    def get_std_from_distribution(label):
        mean = np.matmul(label.numpy().squeeze(), np.arange(30, 230)) / np.sum(label.numpy().squeeze())
        std = np.sqrt(np.matmul(np.square(np.arange(30, 230) - mean), label.numpy().squeeze()) / 1)
        return std
    for ex1, ex2 in parsed_dataset.take(10):
        plt.plot(ex2.numpy().squeeze())
        plt.plot(ex1[1].numpy().squeeze())
        plt.legend(['oracle', 'rt'])
        plt.title(get_std_from_distribution(ex2))
        plt.show()
    parsed_dataset = parsed_dataset.filter(lambda rt, oracle_pred: get_std_from_distribution(oracle_pred) <= 5)
    parsed_dataset = parsed_dataset.map(lambda rt, oracle_pred: finalize_label(rt, oracle_pred, True))
    parsed_dataset = parsed_dataset.map(lambda features, label: join_features(features, label, True))
    # parsed_dataset = parsed_dataset.map(lambda features, label: expand_features_dims(features, label, True))
    for ex1, ex2 in parsed_dataset.take(5):
        pprint(ex1)
        pprint(ex2)
    parsed_dataset = parsed_dataset.batch(32)
    return parsed_dataset


def load_new_user_data(data_dir: str = '../data/new_user', test_size: float = 0.2):
    tfrecord_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecords')]

    train_paths = tfrecord_paths[:int(len(tfrecord_paths) * (1 - test_size))]
    test_paths = tfrecord_paths[int(len(tfrecord_paths) * (1 - test_size)):]

    train_dataset = create_and_parse_dataset(train_paths)
    test_dataset = create_and_parse_dataset(test_paths)

    # for elem in train_dataset.as_numpy_iterator():
    #     pprint(elem)
    #     print(tf.math.reduce_min(elem['ppg']).numpy(), tf.math.reduce_max(elem['ppg']).numpy())
    #     break
    return train_dataset, test_dataset


if __name__ == '__main__':
    train_ds, test_ds = load_new_user_data()

    with one_device_strategy.scope():
        rt_model = get_model('tcn', 'personalization', (CONFIG['sample_size'], 4))
        latest = tf.train.latest_checkpoint(rt_model_path)
        rt_model.load_weights(latest).expect_partial()

    print(rt_model.summary())
    rt_model.evaluate(test_ds, steps=100)
    # freeze layers until tcn_3_conv
    for i, layer in enumerate(rt_model.layers):
        if i < 51:
            layer.trainable = False
        else:
            layer.trainable = True
    # TODO add fill zeros
    # TODO add early stop callback
    # TODO add tensorboard callback
    # TODO have a specific validation dataset
    # Evaluate regular models separately, then retrained one
    rt_model.fit(train_ds, epochs=100, validation_data=test_ds, steps_per_epoch=500, validation_steps=100)
