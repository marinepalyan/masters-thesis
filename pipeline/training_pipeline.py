import argparse
import json
import os
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import tensorflow as tf

import itertools
from datetime import datetime
from models import get_model, MODELS
from pipeline.transforms import *

TOTAL_NUM_OF_USERS = 15
CONFIG = {
    "use_ppg_filter": True,
    "choose_label": "last",
    "model_type": "classification",
    "distribution": "one_hot",
}

def set_config(config_file):
    global CONFIG
    with open(config_file) as f:
        CONFIG = json.load(f)


def build_dataset(ds, transforms, training=False, **CONFIG):
    ds = ds.cache().repeat()
    for transform in transforms[0]:
        ds = ds.map(lambda x: transform(x, training=training, **CONFIG))

    for transform in transforms[1]:
        ds = ds.map(lambda x, y: transform(x, y, training=training, **CONFIG))

    return ds


def prepare_data(input_files: List, test_size: float, **CONFIG):
    test_users_size = 1
    train_dataset = tf.data.TFRecordDataset(input_files[:-test_users_size])
    test_dataset = tf.data.TFRecordDataset(input_files[-test_users_size:])
    transforms = [
        [
            parse_example,
            apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg', 'heart_rate'], func=tf.sparse.to_dense),
            sample_dataset,
            separate_features_label],
        [
            apply_ppg_filter,
            standardize_ppg,
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
    train_ds = build_dataset(train_dataset, transforms, training=True, **CONFIG)
    test_ds = build_dataset(test_dataset, transforms, training=False, **CONFIG)

    train_ds = train_ds.shuffle(100).batch(32)
    test_ds = test_ds.batch(32)
    return train_ds, test_ds


# Define the main function
def main(input_files: List, test_size: float):
    train_ds, test_ds = prepare_data(input_files, test_size)
    main_features = [CONFIG['model_name'], CONFIG['model_type'], CONFIG['choose_label']]
    if CONFIG['distribution'] is not None:
        main_features.append(CONFIG['distribution'])
    if CONFIG['use_ppg_filter']:
        main_features.append('ppg_filter')
    main_work_dir = os.path.join("../logs", datetime.now().strftime("%Y%m%d"), *main_features)
    
    os.makedirs(main_work_dir, exist_ok=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=main_work_dir)
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG['patience'],
                                                        start_from_epoch=50)
    input_shape = (CONFIG['sample_size'], 4)
    # Create the TensorFlow model and compile it
    print(CONFIG['device'])
    one_device_strategy = tf.distribute.OneDeviceStrategy(device=CONFIG['device'])
    print(tf.config.list_physical_devices())
    with one_device_strategy.scope():
        model = get_model(CONFIG['model_name'], CONFIG['model_type'], input_shape)
    # Train the model on the transformed dataset
    model.fit(train_ds, steps_per_epoch=CONFIG['steps_per_epoch'], epochs=CONFIG['epochs'],
                  validation_data=test_ds, validation_steps=CONFIG['validation_steps'],
                  callbacks=[tensorboard_callback, early_stop_callback])
    model_name = '_'.join(main_features)
    save_path = os.path.join(main_work_dir, f"{model_name}.ckpt")
    model.save_weights(save_path)
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
        'model_type': ['regression',],
        'model_name': MODELS.keys(),
        'distribution': [None, 'one_hot', 'gaussian', 'cauchy'],
        'choose_label': ['last', 'middle']
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
