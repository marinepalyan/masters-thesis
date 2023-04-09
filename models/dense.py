import tensorflow as tf


class DenseModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(DenseModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


def create_model():
    model = DenseModel((1500, 4))
    model.compile(optimizer='adam', loss='mse')
    model.build((1500, 4))
    print(model.summary())
    return model
