import os

import tensorflow as tf

from pipeline.training_pipeline import apply_to_keys

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
    # mock activity feature with all zeros
    activity = tf.zeros_like(heart_rate)
    return {
        'heart_rate': heart_rate,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'activity': activity,
        'ppg': ppg,
        'ppg1': example['sensor_readings.105.slot1.pd1'],
        'ppg2': example['sensor_readings.105.slot1.pd2'],
        'ppg3': example['sensor_readings.105.slot1.pd3'],
    }


def load_new_user_data(data_dir: str = '../data/new_user'):
    tfrecord_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecords')]
    raw_dataset = tf.data.TFRecordDataset(tfrecord_paths)

    parsed_dataset = raw_dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))
    parsed_dataset = parsed_dataset.map(
        lambda x: apply_to_keys(keys=feature_description.keys(), func=tf.sparse.to_dense)(x, True))
    parsed_dataset = parsed_dataset.map(format_input)

    # for elem in parsed_dataset.as_numpy_iterator():
    #     pprint(elem)
    #     print(tf.math.reduce_min(elem['ppg']).numpy(), tf.math.reduce_max(elem['ppg']).numpy())
    #     break
    return parsed_dataset
