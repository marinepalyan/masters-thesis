import tensorflow as tf
from keras.losses import mean_absolute_error

HR_GRID = list(range(30, 230, 1))


class WeightedMAE(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='weighted_mae', **kwargs):
        super(WeightedMAE, self).__init__(mean_absolute_error, name=name, **kwargs)
        self.averaging_weights = tf.constant(HR_GRID, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_float = tf.cast(y_pred, tf.float32)
        y_true_reshaped = tf.reshape(y_true_float, tf.shape(y_pred_float))
        y_true_label = tf.tensordot(y_true_reshaped, self.averaging_weights, axes=1)
        y_pred_label = tf.tensordot(y_pred_float, self.averaging_weights, axes=1)
        return super().update_state(y_true_label, y_pred_label, sample_weight=sample_weight)
