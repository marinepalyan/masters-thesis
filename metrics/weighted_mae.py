import tensorflow as tf

HR_GRID = list(range(30, 230, 1))


class WeightedMAE(tf.keras.metrics.Metric):
    def __init__(self, name='weighted_mae', **kwargs):
        super(WeightedMAE, self).__init__(name=name, **kwargs)
        self.total = []
        self.count = []
        self.averaging_weights = tf.constant(HR_GRID, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        y_true = tf.tensordot(y_true, self.averaging_weights, axes=1)
        y_pred = tf.tensordot(y_pred, self.averaging_weights, axes=1)
        mae = tf.math.abs(y_true - y_pred)
        mae = tf.squeeze(mae)
        self.total.append(mae)
        count = tf.cast(tf.size(y_true), tf.float32)
        self.count.append(count)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.)
        self.count.assign(0.)
