import argparse
import logging
import os
import pickle

import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

BVP_SR = 64
ACC_SR = 32
HR_SR = 0.5
ACTIVITY_SR = 4
RESAMPLING_SR = 25  # Use 25Hz as the sampling rate
TOTAL_NUM_OF_USERS = 15


def resample(df: pd.DataFrame, new_sr: int, method: str = "linear") -> pd.DataFrame:
    """Resample and linearly interpolate the data to the new sampling rate.

    Args:
        df: dataframe with timestamp as index
        new_sr: new sampling rate
        method: interpolation method

    Returns:
        resampled dataframe
    """
    # Resample
    df = df.resample(f"{1 / new_sr}S").interpolate(method=method).dropna()
    return df


def save_tfrecord(data: pd.DataFrame, tfrecord_path: str) -> None:
    """Save the data as TFRecord.

    Args:
        data: dataframe with timestamp as index
        tfrecord_path: save path
    """
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        # Save entire dataset as one example
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "acc_x": tf.train.Feature(float_list=tf.train.FloatList(value=data["acc_x"].values)),
                    "acc_y": tf.train.Feature(float_list=tf.train.FloatList(value=data["acc_y"].values)),
                    "acc_z": tf.train.Feature(float_list=tf.train.FloatList(value=data["acc_z"].values)),
                    "ppg": tf.train.Feature(float_list=tf.train.FloatList(value=data["ppg"].values)),
                    "activity": tf.train.Feature(int64_list=tf.train.Int64List(value=data["activity"].values)),
                    "heart_rate": tf.train.Feature(float_list=tf.train.FloatList(value=data["label"].values)),
                }
            )
        )
        writer.write(example.SerializeToString())


def save_output(data: pd.DataFrame, output_dir: str) -> None:
    """Save the data as CSV and TFRecord.

    Args:
        data: dataframe with timestamp as index
        output_dir: save path
    """
    # Save the data
    output_csv_path = os.path.join(output_dir, f"S{user_no}.csv")
    data.to_csv(output_csv_path, index=False)

    # Save as TFRecord as well
    output_tfrecord_path = os.path.join(output_dir, f"S{user_no}.tfrecord")
    save_tfrecord(data, output_tfrecord_path)
    # The code below doesn't work
    # pd2tf(data, output_dir)


def preprocess_user_data(user_no: int, output_dir: str) -> None:
    """Preprocess the data for a single user and save results in the output directory.

    Args:
        user_no: user index
        output_dir: output directory
    """
    # Read in the data
    path = f"../data/S{user_no}/S{user_no}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin-1")

    activity = pd.DataFrame(data["activity"], columns=["activity"]).astype(int)
    label = pd.DataFrame(data["label"], columns=["label"])

    signal = pd.DataFrame(data["signal"])
    acc_df = pd.DataFrame(signal["wrist"].ACC, columns=["acc_x", "acc_y", "acc_z"]).reset_index(drop=True)
    ppg_df = pd.DataFrame(signal["wrist"].BVP, columns=["ppg"]).reset_index(drop=True)

    # Add timestamp to the data
    acc_df["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(acc_df), freq=f"{1 / ACC_SR}S")
    ppg_df["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(ppg_df), freq=f"{1 / BVP_SR}S")
    label["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(label), freq=f"{1 / HR_SR}S")
    activity["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(activity), freq=f"{1 / ACTIVITY_SR}S")

    # Set this column as index
    acc_df.set_index("timestamp", inplace=True)
    ppg_df.set_index("timestamp", inplace=True)
    label.set_index("timestamp", inplace=True)
    activity.set_index("timestamp", inplace=True)

    # Resample the data
    acc_df = resample(acc_df, RESAMPLING_SR)
    ppg_df = resample(ppg_df, RESAMPLING_SR)
    hr_df = resample(label, RESAMPLING_SR)
    activity_df = resample(activity, RESAMPLING_SR, method="nearest").astype(int)

    # Take data points only with these timestamps
    min_sample_time = min(acc_df.index[-1], ppg_df.index[-1], hr_df.index[-1], activity_df.index[-1])
    acc_df = acc_df[:min_sample_time]
    ppg_df = ppg_df[:min_sample_time]
    hr_df = hr_df[:min_sample_time]
    activity_df = activity_df[:min_sample_time]
    assert len(acc_df) == len(ppg_df) == len(hr_df) == len(activity_df)

    # join the data
    data = pd.concat([acc_df, ppg_df, hr_df, activity_df], axis=1)
    save_output(data, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--users', '-u', nargs='+', type=int, default=range(1, TOTAL_NUM_OF_USERS + 1))
    parser.add_argument('--output_dir', '-o', type=str, default="../data/processed/")
    args = parser.parse_args()
    for user_no in args.users:
        logger.info(f"Processing user {user_no}...")
        preprocess_user_data(user_no, args.output_dir)
