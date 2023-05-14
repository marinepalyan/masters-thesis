import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from typing import List

import tensorflow as tf

from pipeline.transforms import apply_to_keys

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


def get_formatted_dataset(input_files: List, training: bool, **CONFIG):
    raw_dataset = tf.data.TFRecordDataset(input_files)

    parsed_dataset = raw_dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
    parsed_dataset = parsed_dataset.map(
        lambda x: apply_to_keys(keys=feature_description.keys(), func=tf.sparse.to_dense)(x, True, **CONFIG))
    parsed_dataset = parsed_dataset.map(format_input)
    lengths = []
    # for elem in parsed_dataset.as_numpy_iterator():
    #     lengths.append(len(elem['acc_x']))
    #     # break
    # print(sum(lengths) / (25 * 3600)) # 16.733533333333334 hours (44 minutes)
    return parsed_dataset
