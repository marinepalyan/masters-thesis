import tensorflow as tf
def create_model_():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model


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
