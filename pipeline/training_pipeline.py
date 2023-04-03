import tensorflow as tf
import argparse
from typing import List, Callable
import numpy as np

SAMPLE_SIZE = 1500
input_shape = (SAMPLE_SIZE, 4)
epochs = 10
TOTAL_NUM_OF_USERS = 15


def apply_to_keys(keys: List[str], func: Callable):
    def apply_to_keys_fn(dataset):
        for key in keys:
            dataset[key] = func(dataset[key])
        return dataset
    return apply_to_keys_fn


# specify to do this only when training
# how much future is useful? e.g. 1-minute data into the future
# after a while start losing past data and results get worse
# use not the last label, but in the middle, or 75th percentile
def fill_zeros(features, training: bool = True):
    if not training:
        return features
    prob_to_run = tf.random.uniform([], maxval=4, dtype=tf.int32)
    # If the number is 0, then run the transform
    if prob_to_run != 0:
        return features
    # Generate a random number between 0 and 1250 inclusive
    n = tf.random.uniform([], maxval=1250, dtype=tf.int32)
    # Assign the first n values to zeros
    def slice_zeros_wrapper(column):
        return tf.concat([tf.zeros([n]), tf.slice(column, [n], [SAMPLE_SIZE - n])], axis=0)
    features = apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg'],
                                func=slice_zeros_wrapper)(features)
    features = tf.concat(features, axis=0)
    return features


def choose_label(label):
    return tf.slice(label, [-1], [1])


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


def sample_dataset(dataset):
    # TODO fix maxval
    # print(tf.shape(dataset['acc_x']))
    n = tf.random.uniform([], maxval=10000 - SAMPLE_SIZE - 1, dtype=tf.int32)
    def slice_wrapper(column):
        return tf.slice(column, [n], [SAMPLE_SIZE])
    dataset = dataset.repeat(1000).map(apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg', 'heart_rate'],
                                                 func=slice_wrapper))
    dataset = tf.concat(dataset, axis=0)
    return dataset


def separate_features_label(dataset):
    return dataset
    print(dataset['acc_x'])
    print(dataset['heart_rate'])
    features = dataset['acc_x']
    label = dataset['heart_rate']
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

    # Map each example to its features and label
    dataset = dataset.map(parse_example)
    dataset = dataset.map(apply_to_keys(keys=['acc_x', 'acc_y', 'acc_z', 'ppg',
                                              'activity', 'heart_rate'], func=tf.sparse.to_dense))

    dataset = sample_dataset(dataset)
    for e in dataset.as_numpy_iterator():
        print(e)
        break
    # Expected begin[0] == 0 (got 681) and size[0] == 0 (got 1500) when input.dim_size(0) == 0
    features, label = dataset.map(separate_features_label)
    features = features.map(fill_zeros)
    for e in features.as_numpy_iterator():
        print(e)
        break
    label = label.map(choose_label)
    for e in label.as_numpy_iterator():
        print(e)
        break
    dataset = tf.data.Dataset.zip((features, label))
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
