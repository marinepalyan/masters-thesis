from keras.losses import categorical_crossentropy


def my_loss_fn(y_true, y_pred):
    return categorical_crossentropy(y_true[:, 1, :], y_pred)
