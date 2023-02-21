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
                        "ACC_x": tf.train.Feature(float_list=tf.train.FloatList(value=[row.ACC_x])),
                        "ACC_y": tf.train.Feature(float_list=tf.train.FloatList(value=[row.ACC_y])),
                        "ACC_z": tf.train.Feature(float_list=tf.train.FloatList(value=[row.ACC_z])),
                        "PPG": tf.train.Feature(float_list=tf.train.FloatList(value=[row.PPG])),
                        "HR": tf.train.Feature(float_list=tf.train.FloatList(value=[row.Label])),
                    }
                )
            )
            writer.write(example.SerializeToString())


def preprocess_user_data(user_no):
    # Read in the data
    path = f"../data/S{user_no}/S{user_no}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin-1")

    # TODO figure out activity sampling rate
    activity = pd.DataFrame(data["activity"]).astype(int)
    activity.columns = ["Activity"]

    label = pd.DataFrame(data["label"])
    label.columns = ["Label"]

    signal = pd.DataFrame(data["signal"])

    ACC = pd.DataFrame(signal["wrist"].ACC)
    ACC.columns = ["ACC_x", "ACC_y", "ACC_z"]
    ACC.reset_index(drop=True, inplace=True)

    PPG = pd.DataFrame(signal["wrist"].BVP)
    PPG.columns = ["PPG"]
    PPG.reset_index(drop=True, inplace=True)

    # Add timestamp to the data
    ACC["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(ACC), freq=f"{1 / ACC_SR}S")
    PPG["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(PPG), freq=f"{1 / BVP_SR}S")
    label["timestamp"] = pd.timedelta_range(start="00:00:00", periods=len(label), freq=f"{1 / HR_SR}S")

    # Set this column as index
    ACC.set_index("timestamp", inplace=True)
    PPG.set_index("timestamp", inplace=True)
    label.set_index("timestamp", inplace=True)

    # Resample the data
    ACC = resample(ACC, SECOND_SR)
    PPG = resample(PPG, SECOND_SR)
    HR = resample(label, SECOND_SR)

    logger.info(f"Length of ACC: {len(ACC)}")
    logger.info(f"Length of PPG: {len(PPG)}")
    logger.info(f"Length of HR: {len(HR)}")

    # Take data points only with these timestamps
    min_sample_time = min(ACC.index[-1], PPG.index[-1], HR.index[-1])
    timestamps = pd.timedelta_range(start="00:00:00", end=min_sample_time, freq=f"{1 / NEW_SR}S")
    ACC = ACC.loc[timestamps]
    PPG = PPG.loc[timestamps]
    HR = HR.loc[timestamps]

    logger.info(f"Length of ACC: {len(ACC)}")
    logger.info(f"Length of PPG: {len(PPG)}")
    logger.info(f"Length of HR: {len(HR)}")

    # join the data
    data = pd.concat([ACC, PPG, HR], axis=1)
    data.head()

    # Save the data
    data.to_csv(f"../data/processed/S{user_no}.csv", index=False)

    # Save as TFRecord as well
    tfrecord_path = f"../data/processed/S{user_no}.tfrecord"
    save_tfrecord(data, tfrecord_path)
    # The code below doesn't work
    # tfrecord_folder = f"../data/processed/S{user_no}/"
    # os.makedirs(tfrecord_folder, exist_ok=True)
    # pd2tf(data, tfrecord_folder)


for user_no in range(1, 16):
    logger.info(f"Processing user {user_no}...")
    preprocess_user_data(user_no)
