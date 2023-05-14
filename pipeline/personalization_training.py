import os

import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from typing import List

import tensorflow as tf

import numpy as np

from models import get_model
from pipeline.training_pipeline import prepare_data
from pipeline.transforms import *
from data_processing.preprocess_new_user_data import get_formatted_dataset

CONFIG = {
    "sample_size": 1565,
    "device": "/gpu:0",
    "use_augmentation": True,
    "patience": 10,
    "use_ppg_filter": True,
    "choose_label": "last",
    "std_threshold": 2.5,
    "batch_size": 32,
}
models_path = '/home/marine/learning/masters-thesis/logs/logs/20230508/tcn/classification'
oracle_model_path = f'{models_path}/middle/one_hot/ppg_filter/'
rt_model_path = f'{models_path}/last/one_hot/ppg_filter/'
save_path = f'{models_path}/retrained'
os.makedirs(save_path, exist_ok=True)

one_device_strategy = tf.distribute.OneDeviceStrategy(device=CONFIG['device'])
print(tf.config.list_physical_devices())
with one_device_strategy.scope():
    oracle_model = get_model('tcn', 'classification', (CONFIG['sample_size'], 4))
    latest = tf.train.latest_checkpoint(oracle_model_path)
    oracle_model.load_weights(latest).expect_partial()
    rt_model = get_model('tcn', 'classification', (CONFIG['sample_size'], 4))
    latest = tf.train.latest_checkpoint(rt_model_path)
    rt_model.load_weights(latest).expect_partial()

CONFIG['oracle_model'] = oracle_model
CONFIG['rt_model'] = rt_model


def create_and_parse_dataset(input_files: List, training: bool):
    parsed_dataset = get_formatted_dataset(input_files, training, **CONFIG)
    parsed_dataset = parsed_dataset.cache().repeat()

    parsed_dataset = parsed_dataset.map(lambda x: double_sample_dataset(x, True, **CONFIG))
    parsed_dataset = parsed_dataset.map(lambda rt, oracle: (separate_features_label(rt, True, **CONFIG),
                                                            separate_features_label(oracle, True, **CONFIG)))
    transforms = [
        apply_ppg_filter,
        standardize_ppg
    ]
    for transform in transforms:
        parsed_dataset = parsed_dataset.map(lambda rt, oracle: (transform(*rt, True, **CONFIG),
                                                                transform(*oracle, True, **CONFIG)))
    rt_transforms = [
        choose_label,
        one_hot_label
    ]
    if CONFIG['use_augmentation']:
        rt_transforms.append(fill_zeros)
    for transform in rt_transforms:
        parsed_dataset = parsed_dataset.map(lambda rt, oracle: (transform(*rt, True, **CONFIG), oracle))

    oracle_transforms = [
        join_features,
        expand_features_dims,
        predict_label,
    ]
    for transform in oracle_transforms:
        parsed_dataset = parsed_dataset.map(lambda rt, oracle: (rt, transform(*oracle, True, **CONFIG)))

    parsed_dataset = parsed_dataset.filter(lambda rt, oracle_pred: filter_by_label(oracle_pred,
                                                                                   CONFIG['std_threshold']))
    transforms = [
        finalize_label,
        join_features,
        # expand_features_dims,
    ]
    for transform in transforms:
        parsed_dataset = parsed_dataset.map(lambda features, label: (transform(features, label, True, **CONFIG)))
    parsed_dataset = parsed_dataset.batch(CONFIG['batch_size'])
    return parsed_dataset


def load_new_user_data(data_dir: str = '../data/new_user', test_size: float = 0.2):
    tfrecord_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecords')]

    train_paths = tfrecord_paths[:int(len(tfrecord_paths) * (1 - test_size))]
    test_paths = tfrecord_paths[int(len(tfrecord_paths) * (1 - test_size)):]

    train_dataset = create_and_parse_dataset(train_paths, True)
    test_dataset = create_and_parse_dataset(test_paths, False)
    return train_dataset, test_dataset


if __name__ == '__main__':
    new_train_ds, new_test_ds = load_new_user_data()
    input_files = [f'../data/processed/S{user}.tfrecord' for user in range(13, 16)]
    train_ds, test_ds = prepare_data(input_files, 0.33)
    validation_ds = test_ds.take(100)

    with one_device_strategy.scope():
        retrain_rt_model = get_model('tcn', 'personalization', (CONFIG['sample_size'], 4))
        latest = tf.train.latest_checkpoint(rt_model_path)
        retrain_rt_model.load_weights(latest).expect_partial()

    # print("Evaluating Oracle Model")
    # oracle_model.evaluate(validation_ds)
    print("Evaluating RT Model")
    rt_model.evaluate(validation_ds)
    print("Retraining RT Model")
    # freeze layers until tcn_3_conv
    for i, layer in enumerate(retrain_rt_model.layers):
        if i < 51:
            layer.trainable = False
        else:
            layer.trainable = True
    # Evaluate regular models separately, then retrained one
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="../logs/personalization/")
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG['patience'],
                                                        start_from_epoch=20)
    retrain_rt_model.fit(new_train_ds, epochs=150, validation_data=new_test_ds,
                         steps_per_epoch=300, validation_steps=50,
                         callbacks=[tensorboard_callback, early_stop_callback])
    retrain_rt_model.save_weights(os.path.join(save_path, 'retrained_model.h5'))
    print("Evaluating Retrained RT Model")
    retrain_rt_model.evaluate(validation_ds)
