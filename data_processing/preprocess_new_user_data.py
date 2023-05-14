import os

import keras

from metrics.weighted_mae import WeightedMAE
from metrics.weighted_mse import WeightedMSE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pprint import pprint
from typing import List

import tensorflow as tf
from matplotlib import pyplot as plt

import numpy as np

from models import get_model
from pipeline.training_pipeline import (apply_to_keys, separate_features_label, apply_ppg_filter,
                                        standardize_ppg, join_features, choose_label, one_hot_label,
                                        fill_zeros, prepare_data)


CONFIG = {
    "sample_size": 1565,
    "device": "/gpu:0",
    "use_augmentation": True,
    "patience": 10,
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
    rt_model = get_model('tcn', 'classification', (CONFIG['sample_size'], 4))
    latest = tf.train.latest_checkpoint(rt_model_path)
    rt_model.load_weights(latest).expect_partial()


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
    label = tf.reshape(label, (1, 200))
    # join label and oracle_pred
    new_label = tf.concat([label, oracle_pred], axis=0)
    return features, new_label


def create_and_parse_dataset(input_files: List, training: bool):
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
    if CONFIG['use_augmentation']:
        parsed_dataset = parsed_dataset.map(lambda rt, oracle: (fill_zeros(*rt, training), oracle))
    # for rt, oracle in parsed_dataset.take(5):
    #     from matplotlib import pyplot as plt
    #     plt.plot(rt[0]['ppg'])
    #     plt.plot(oracle[0]['ppg'])
    #     plt.legend(['rt', 'oracle'])
    #     plt.show()
    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (rt, join_features(*oracle, True)))
    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (rt, expand_features_dims(*oracle, True)))


    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (predict_label(rt, oracle, True)))
    # for ex1, ex2 in parsed_dataset.take(10):
    #     label, pred = ex2
    #     pred_vector = pred.numpy().squeeze()
    #     plt.plot(label['heart_rate'].numpy().squeeze())
    #     plt.title(f"Pred: {(pred.numpy().squeeze() * range(30, 230)).sum()}")
    #     plt.show()

    parsed_dataset = parsed_dataset.map(lambda rt, oracle_pred: (choose_label(*rt, True), oracle_pred))
    for ex1, ex2 in parsed_dataset.take(5):
        print(ex2.numpy().squeeze())
        print(ex1[1].numpy().squeeze())
    parsed_dataset = parsed_dataset.map(lambda rt, oracle_pred: (one_hot_label(*rt, True), oracle_pred))

    def get_std_from_distribution(label):
        # create tensor from np.arange(30, 230)
        dist_values = tf.constant(np.arange(30, 230), dtype=tf.float32)
        mean = tf.reduce_sum(tf.math.multiply(dist_values, label))
        std = tf.sqrt(tf.reduce_sum(tf.math.multiply(tf.square(dist_values - mean), label)))
        return std
    # for ex1, ex2 in parsed_dataset.take(10):
    #     get_std_from_distribution(tf.squeeze(ex2))
    #     plt.plot(ex2.numpy().squeeze())
    #     plt.plot(ex1[1].numpy().squeeze())
    #     plt.legend(['oracle', 'rt'])
    #     plt.title(get_std_from_distribution(tf.squeeze(ex2)).numpy())
    #     plt.show()
    parsed_dataset = parsed_dataset.filter(lambda rt, oracle_pred: get_std_from_distribution(oracle_pred) <= 2.5)
    parsed_dataset = parsed_dataset.map(lambda rt, oracle_pred: finalize_label(rt, oracle_pred, True))
    parsed_dataset = parsed_dataset.map(lambda features, label: join_features(features, label, True))
    parsed_dataset = parsed_dataset.map(lambda features, label: expand_features_dims(features, label, True))
    for ex1, ex2 in parsed_dataset.take(5):
        pprint(ex1)
        pprint(ex2)
    parsed_dataset = parsed_dataset.batch(32)
    return parsed_dataset


def load_new_user_data(data_dir: str = '../data/new_user', test_size: float = 0.2):
    tfrecord_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecords')]

    train_paths = tfrecord_paths[:int(len(tfrecord_paths) * (1 - test_size))]
    test_paths = tfrecord_paths[int(len(tfrecord_paths) * (1 - test_size)):]

    train_dataset = create_and_parse_dataset(train_paths, True)
    test_dataset = create_and_parse_dataset(test_paths, False)

    # for elem in train_dataset.as_numpy_iterator():
    #     pprint(elem)
    #     print(tf.math.reduce_min(elem['ppg']).numpy(), tf.math.reduce_max(elem['ppg']).numpy())
    #     break
    return train_dataset, test_dataset


if __name__ == '__main__':
    new_train_ds, new_test_ds = load_new_user_data()
    input_files = [f'../data/processed/S{user}.tfrecord' for user in range(13, 16)]
    train_ds, test_ds = prepare_data(input_files, 0.33)
    # for features, label in new_train_ds.take(10):
    #     pprint(features)
    #     pprint(label)
    #     joined_features, _ = join_features(features, label, True)
    #     joined_features = tf.reshape(joined_features, (1, joined_features.shape[0], joined_features.shape[1]))
    #     rt_pred = rt_model(joined_features)
    #     plt.plot(features['acc_x'], label='acc_x')
    #     plt.plot(features['acc_y'], label='acc_y')
    #     plt.plot(features['acc_z'], label='acc_z')
    #     plt.legend()
    #     plt.title(f"New \n Label: {np.argmax(label.numpy()[0]) + 30}, prediction: {(label.numpy()[1] * np.arange(30, 230, 1)).sum()} \n"
    #               f"RT prediction: {(rt_pred.numpy()[0] * np.arange(30, 230, 1)).sum()}")
    #     plt.show()
    # for features, label in train_ds.take(30):
    #     joined_features, _ = join_features(features, label, True)
    #     joined_features = tf.reshape(joined_features, (1, joined_features.shape[0], joined_features.shape[1]))
    #     oracle_pred = oracle_model(joined_features)
    #     rt_pred = rt_model(joined_features)
    #     plt.plot(features['acc_x'], label='acc_x')
    #     plt.plot(features['acc_y'], label='acc_y')
    #     plt.plot(features['acc_z'], label='acc_z')
    #     plt.legend()
    #     plt.title(
    #         f"Train \n Label: {np.argmax(label.numpy()) + 30}, prediction: {(rt_pred.numpy()[0] * np.arange(30, 230, 1)).sum()}")
    #     plt.show()
    validation_ds = test_ds.take(100)

    with one_device_strategy.scope():
        rt_model = get_model('tcn', 'classification', (CONFIG['sample_size'], 4))
        latest = tf.train.latest_checkpoint(rt_model_path)
        rt_model.load_weights(latest).expect_partial()

        oracle_model = get_model('tcn', 'classification', (CONFIG['sample_size'], 4))
        latest = tf.train.latest_checkpoint(oracle_model_path)
        oracle_model.load_weights(latest).expect_partial()

    print(rt_model.summary())
    print("Evaluating Oracle Model")
    oracle_model.evaluate(validation_ds)
    print("Evaluating RT Model")
    rt_model.evaluate(validation_ds)
    print("Retraining RT Model")
    # freeze layers until tcn_3_conv
    for i, layer in enumerate(rt_model.layers):
        if i < 51:
            layer.trainable = False
        else:
            layer.trainable = True
    # Evaluate regular models separately, then retrained one
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="../logs/personalization/")
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG['patience'],
                                                        start_from_epoch=20)
    rt_model.fit(train_ds, epochs=150, validation_data=test_ds, steps_per_epoch=300, validation_steps=50,
                 callbacks=[tensorboard_callback, early_stop_callback])
    rt_model.save_weights(os.path.join(rt_model_path, 'retrained_model.h5'))
    print("Evaluating Retrained RT Model")
    rt_model.evaluate(validation_ds)
