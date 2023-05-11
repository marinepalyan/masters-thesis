import tensorflow as tf
from keras.losses import mean_absolute_error, categorical_crossentropy

HR_GRID = list(range(30, 230, 1))

def my_loss_fn(y_true, y_pred):
    return categorical_crossentropy(y_true[:, 1, :], y_pred)


# class TwoLabelLoss(tf.keras.losses.Loss):
#     def __init__(self, name='two_label_loss'):
#         super(TwoLabelLoss, self).__init__(name=name, reduction=categorical_crossentropy)
#
#     def __call__(self, y_true, y_pred, **kwargs):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
#         print(tf.shape(y_true))
#         print(y_true)
#         print(tf.shape(y_pred))
#         print(y_pred)
#         # y_true shape is (2, 200)
#         # need to take the first element and have size (1, 200)
#         y_true = y_true[1, :]
#         print(tf.shape(y_true))
#         print(y_true)
#         y_true = tf.reshape(y_true, tf.shape(y_pred))
#         print(tf.shape(y_true))
#         print(y_true)
#         print(f"CE loss: {tf.keras.losses.categorical_crossentropy(y_true, y_pred)}")
#         return super().__call__(y_true, y_pred)
