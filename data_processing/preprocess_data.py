import argparse
import logging
import os
import pickle

import pandas as pd
import tensorflow as tf
from pandas_tfrecords import pd2tf

logger = logging.getLogger(__name__)

BVP_SR = 64
ACC_SR = 32
HR_SR = 0.5
SECOND_SR = 1000
NEW_SR = 25  # Use 25Hz as the sampling rate
TOTAL_NUM_OF_USERS = 15


# TODO sliding window approach?


def resample(df, new_sr):
    df = df.resample(f"{1 / new_sr}S").mean()
    df = df.interpolate(method="linear")
    return df


def save_tfrecord(data, tfrecord_path):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for row in data.itertuples():
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "acc_x": tf.train.Feature(float_list=tf.train.FloatList(value=[row.acc_x])),
                        "acc_y": tf.train.Feature(float_list=tf.train.FloatList(value=[row.acc_y])),
                        "acc_z": tf.train.Feature(float_list=tf.train.FloatList(value=[row.acc_z])),
                        "ppg": tf.train.Feature(float_list=tf.train.FloatList(value=[row.ppg])),
                        "heart_rate": tf.train.Feature(float_list=tf.train.FloatList(value=[row.label])),
                    }
                )
            )
            writer.write(example.SerializeToString())


def save_output(data, output_dir):
    # Save the data
    output_csv_path = os.path.join(output_dir, f"S{user_no}.csv")
    data.to_csv(output_csv_path, index=False)

    # Save as TFRecord as well
    output_tfrecord_path = os.path.join(output_dir, f"S{user_no}.tfrecord")
    save_tfrecord(data, output_tfrecord_path)
    # The code below doesn't work
    # pd2tf(data, output_dir)


def preprocess_user_data(user_no, output_dir):
    # Read in the data
    path = f"../data/S{user_no}/S{user_no}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin-1")

    # TODO figure out activity sampling rate
    activity = pd.DataFrame(data["activity"], columns=["activity"]).astype(int)
    label = pd.DataFrame(data["label"], columns=["label"])

    signal = pd.DataFrame(data["signal"])
    acc_df = pd.DataFrame(signal["wrist"].ACC, columns=["acc_x", "acc_y", "acc_z"]).reset_index(drop=True)
    ppg_df = pd.DataFrame(signal["wrist"].BVP, columns=["ppg"]).reset_index(drop=True)

    # Add timestamp to the data
    acc_df["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(acc_df), freq=f"{1 / ACC_SR}S")
    ppg_df["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(ppg_df), freq=f"{1 / BVP_SR}S")
    label["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(label), freq=f"{1 / HR_SR}S")

    # Set this column as index
    acc_df.set_index("timestamp", inplace=True)
    ppg_df.set_index("timestamp", inplace=True)
    label.set_index("timestamp", inplace=True)

    # Resample the data
    acc_df = resample(acc_df, SECOND_SR)
    ppg_df = resample(ppg_df, SECOND_SR)
    hr_df = resample(label, SECOND_SR)

    logger.info(f"Length of ACC: {len(acc_df)}")
    logger.info(f"Length of PPG: {len(ppg_df)}")
    logger.info(f"Length of HR: {len(hr_df)}")

    # Take data points only with these timestamps
    min_sample_time = min(acc_df.index[-1], ppg_df.index[-1], hr_df.index[-1])
    timestamps = pd.timedelta_range(start="00:00:00", end=min_sample_time, freq=f"{1 / NEW_SR}S")
    acc_df = acc_df.loc[timestamps]
    ppg_df = ppg_df.loc[timestamps]
    hr_df = hr_df.loc[timestamps]

    logger.info(f"Length of ACC: {len(acc_df)}")
    logger.info(f"Length of PPG: {len(ppg_df)}")
    logger.info(f"Length of HR: {len(hr_df)}")

    # join the data
    data = pd.concat([acc_df, ppg_df, hr_df], axis=1)
    save_output(data, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--users', '-u', nargs='+', type=int, default=range(1, TOTAL_NUM_OF_USERS + 1))
    parser.add_argument('--output_dir', '-o', type=str, default="../data/processed/")
    args = parser.parse_args()
    for user_no in args.users:
        logger.info(f"Processing user {user_no}...")
        preprocess_user_data(user_no, args.output_dir)
