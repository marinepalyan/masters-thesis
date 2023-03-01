import os

import tensorflow as tf

input_shape = (1500, 4)
epochs = 10


def transform_dataset(features, label):
    # Generate a random number between 0 and 1400 inclusive
    n = tf.random.uniform([], maxval=1400, dtype=tf.int32)
    # Assign the first n values to zeros
    features = tf.concat([tf.zeros([n]), features[n:]], axis=0)
    return features, label


# Define a function to parse each example in the TFRecord file
def parse_example(example_proto):
    feature_description = {
        "acc_x": tf.io.FixedLenFeature([], tf.float32),
        "acc_y": tf.io.FixedLenFeature([], tf.float32),
        "acc_z": tf.io.FixedLenFeature([], tf.float32),
        "ppg": tf.io.FixedLenFeature([], tf.float32),
        "activity": tf.io.FixedLenFeature([], tf.int64),
        "heart_rate": tf.io.FixedLenFeature([], tf.float32)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example
    features = [example['acc_x'], example['acc_y'], example['acc_z'], example['ppg']]
    label = example['heart_rate']
    return features, label


def sample_dataset(dataset):
    # take 1500 rows of data
    data_list = dataset.shuffle(10000).take(1500)
    # Extract the features and labels
    features = [data[0] for data in data_list]
    labels = [data[1] for data in data_list]

    # Sample 1500 rows of data
    features = features[:1500]
    last_label = labels[-1]

    # Return the sampled features and final label
    return features, last_label


def sample_dataset(dataset):
    # Create a buffer size for shuffling the dataset
    buffer_size = tf.data.experimental.cardinality(dataset).numpy()

    # Shuffle the dataset and take 1500 rows
    dataset = dataset.shuffle(buffer_size=buffer_size).take(1500)

    # Extract the features and labels from the dataset
    features, labels = [], []
    for feature, label in dataset.as_numpy_iterator():
        features.append(feature)

    # Set the final label as the label of the last row
    last_label = labels[-1]

    # Return the sampled features and final label
    return features, last_label


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
def main():
    # Create a TensorFlow dataset from a TFRecord file
    # Load the TFRecord file into a dataset
    if not os.path.exists('../data/processed/S1.tfrecord'):
        raise FileNotFoundError('TFRecord file not found')

    # Load TFRecord dataset
    dataset = tf.data.TFRecordDataset('../data/processed/S1.tfrecord')

    # Map each example to its features and label
    dataset = dataset.map(parse_example)
    dataset = sample_dataset(dataset)
    dataset = dataset.map(transform_dataset)

    # Create the TensorFlow model and compile it
    model = create_model()
    # Train the model on the transformed dataset
    model.fit(dataset, epochs=epochs)


if __name__ == '__main__':
    main()
