import tensorflow as tf
import argparse
from typing import List, Callable
import numpy as np
from pprint import pprint

SAMPLE_SIZE = 1500
input_shape = (SAMPLE_SIZE, 4)
epochs = 10
TOTAL_NUM_OF_USERS = 15


def apply_to_keys(keys: List[str], func: Callable):
    def apply_to_keys_fn(example):
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
    print(prob_to_run)
    if prob_to_run == 0:
        return features, label
    print("success")
    # Generate a random number between 0 and 1250 inclusive
    n = tf.random.uniform([], maxval=1250, dtype=tf.int32)
    # Assign the first n values to zeros
    def slice_zeros_wrapper(column):
        return tf.concat([tf.zeros([n]), tf.slice(column, [n], [SAMPLE_SIZE - n])], axis=0)
    features = apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg'],
                                func=slice_zeros_wrapper)(features)
    return features, label


def choose_label(features, label):
    return features, label['heart_rate'][-1]


def join_features(features, label):
    features = tf.concat([features['acc_x'], features['acc_y'], features['acc_z'], features['ppg']], axis=0)
    features = tf.reshape(features, input_shape)
    print(type(label))
    print(label)
    label = tf.reshape(label, (1,))
    return features, label


# Define a function to parse each example in the TFRecord file
def parse_example(example_proto):
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


def sample_dataset(example):
    # TODO fix maxval
    start_idx = tf.random.uniform((), minval=0, maxval=tf.shape(example['acc_x'])[0] - SAMPLE_SIZE - 1, dtype=tf.int32)
    # tf.print(f"n: {n}")
    def slice_column(column):
        return column[start_idx:start_idx + SAMPLE_SIZE]
    keys = ['acc_x', 'acc_y', 'acc_z', 'ppg', 'heart_rate']
    example = (apply_to_keys(keys=keys, func=slice_column))(example)
    return example


def separate_features_label(dataset):
    features = dataset.copy()
    label = {"heart_rate": features.pop('heart_rate')}
    return features, label


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model


# Define the main function
def main(input_files: List):
    # Load TFRecord dataset
    dataset = tf.data.TFRecordDataset(input_files)

    dataset = dataset.map(parse_example)
    dataset = dataset.map(apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg',
                                              'activity', 'heart_rate'], func=tf.sparse.to_dense))
    # for e in dataset.as_numpy_iterator():
    #     print(e['acc_x'].shape)
    dataset = dataset.repeat(10000).map(sample_dataset)
    # for i, e in enumerate(dataset.as_numpy_iterator()):
    #     print(e['acc_x'].shape)
    #     if i > 20:
    #         break
    dataset = dataset.map(separate_features_label)
    for features, label in dataset.as_numpy_iterator():
        pprint(label)
        print(features['acc_x'].shape)
        print(label['heart_rate'].shape)
        break
    print('before fill zeros')
    dataset = dataset.map(fill_zeros)
    for e in dataset.as_numpy_iterator():
        pprint(e)
        break
    dataset = dataset.map(choose_label)
    dataset = dataset.map(join_features)
    print("lalalala")
    print(dataset)
    # FIXME this is not working
    # getting an error that the dataset is not iterable
    # slice index -1 of dimension 0 out of bounds.
    for (features, label) in dataset.as_numpy_iterator():
        print(features)
        print(features.shape)
        print(label)
    # dataset = tf.data.Dataset.zip((features, label))
    dataset = dataset.batch(32)
    # bool argument to decide whether to do augmentations: training = False
    # lambda x: fill_zeros(x, training=training)

    # Create the TensorFlow model and compile it
    model = create_model()
    # Train the model on the transformed dataset
    model.fit(dataset, epochs=epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--users', '-u', nargs='+', type=int, default=range(1, TOTAL_NUM_OF_USERS + 1))
    args = parser.parse_args()
    input_files = [f'../data/processed/S{user}.tfrecord' for user in args.users]
    main(input_files)
