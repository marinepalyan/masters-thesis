import tensorflow as tf
from keras.losses import mean_absolute_error, categorical_crossentropy

HR_GRID = list(range(30, 230, 1))


class TwoLabelLoss(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='two_label_loss', **kwargs):
        super(TwoLabelLoss, self).__init__(categorical_crossentropy, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        print(sample_weight)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        print(tf.shape(y_true))
        print(y_true)
        print(tf.shape(y_pred))
        print(y_pred)
        # y_true = y_true[:, 1]
        # print(tf.shape(y_true))
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)