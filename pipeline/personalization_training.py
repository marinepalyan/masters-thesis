import argparse
import json
import tensorflow as tf

from data_processing.preprocess_new_user_data import load_new_user_data
from metrics.weighted_mae import WeightedMAE
from metrics.weighted_mse import WeightedMSE
from models import get_model
from pipeline.training_pipeline import (sample_dataset, separate_features_label, fill_zeros, choose_label,
                                        join_features, build_dataset, dist_label, set_config)

CONFIG = {"sample_size": 1565}


def prepare_data(input_dir: str, test_size: float):
    train_ds, test_ds = load_new_user_data(input_dir, test_size)
    transforms = [
        [
            sample_dataset,
            separate_features_label],
        [
            fill_zeros,
            choose_label,
            join_features
        ]
    ]

    transforms[1].append(dist_label)

    train_ds = build_dataset(train_ds, transforms, training=True)
    test_ds = build_dataset(test_ds, transforms, training=False)

    train_ds = train_ds.shuffle(100).batch(32)
    test_ds = test_ds.batch(32)
    return train_ds, test_ds


def main(input_dir: str, test_size: float, config_path: str):
    with open(config_path, 'r') as f:
        CONFIG = json.load(f)
    input_shape = (CONFIG['sample_size'], 4)
    train_ds, test_ds = prepare_data(input_dir, test_size)
    rt_model_path = '/home/marine/learning/masters-thesis/logs/logs/20230505/' \
                    'tcn/classification/last/gaussian/'
    oracle_model_path = '/home/marine/learning/masters-thesis/logs/logs/20230505/' \
                        'tcn/classification/middle/gaussian/'
    rt_weights = tf.train.latest_checkpoint(rt_model_path)
    print(rt_weights)
    oracle_weights = tf.train.latest_checkpoint(oracle_model_path)
    print(oracle_weights)


    one_device_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(tf.config.list_physical_devices())
    with one_device_strategy.scope():
        student_model = get_model(CONFIG['model_name'], CONFIG['model_type'], input_shape)
        teacher_model = get_model(CONFIG['model_name'], CONFIG['model_type'], input_shape)
        student_model.load_weights(rt_weights)
        teacher_model.load_weights(oracle_weights)

    # teacher_model.fit(train_ds, epochs=CONFIG['epochs'], validation_data=test_ds,
    #                       steps_per_epoch=CONFIG['steps_per_epoch'], validation_steps=CONFIG['validation_steps'])
    preds = teacher_model.predict(train_ds.take(1))
    print(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../data/new_user')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--config_path', type=str, default='../personalisation_config.json')
    args = parser.parse_args()
    main(args.input_dir, args.test_size, args.config_path)
