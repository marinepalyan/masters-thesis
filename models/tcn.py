from typing import Optional, Text, Sequence, Tuple

import tensorflow as tf

EXPECTED_INPUT_SHAPE = (1565, 4)


def _nx1_conv_block(
        filters: int,
        kernel_size: int,
        pool_size: int,
        dilation_rate: int = 1,
        spatial_drop_rate: float = 0.0,
        name: Optional[Text] = None):
    """A nx1 2D conv block with batch norm, pooling and dilation."""

    def f(x):
        x = tf.keras.layers.Conv2D(
            filters, (kernel_size, 1),
            dilation_rate=(dilation_rate, 1),
            padding='valid',
            name=f'{name}_conv')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn')(x)
        if spatial_drop_rate > 0:
            x = tf.keras.layers.SpatialDropout2D(spatial_drop_rate, name=f'{name}_dropout')(x)
        x = tf.keras.layers.ReLU(name=f'{name}_relu')(x)
        if pool_size > 1:
            x = tf.keras.layers.MaxPooling2D((pool_size, 1), name=f'{name}_pool')(x)
        return x

    return f


def cnn_tcn(nr_features: int,
            seq_len: int,
            cnn_layers: Sequence[Tuple[int, int, int]],
            tcn_layers: Sequence[Tuple[int, int]],
            tcn_dilation: int,
            dense_layers: Sequence[int],
            spatial_drop_rate: float = 0.05,
            dense_drop_rate: float = 0.1,
            output_activation: str = 'relu',
            num_of_classes: int = 1):
    input_l = x = tf.keras.layers.Input(shape=(seq_len, nr_features), name='input')
    x = tf.keras.layers.Reshape(target_shape=(seq_len, 1, nr_features))(x)
    for l_idx, (filters, kernel_size, pool_size) in enumerate(cnn_layers):
        name = f'cnn_{l_idx + 1}'
        x = _nx1_conv_block(filters, kernel_size, pool_size, 1, spatial_drop_rate, name)(x)
    dilation_counter = 0
    for l_idx, (filters, kernel_size) in enumerate(tcn_layers):
        name = f'tcn_{l_idx + 1}'
        dilation_rate = tcn_dilation ** dilation_counter
        dilation_counter += 1
        x = _nx1_conv_block(filters, kernel_size, 1, dilation_rate, spatial_drop_rate, name)(x)
    flatten_layer = tf.keras.layers.Flatten()
    x = flatten_layer(x)
    if dense_layers is not None:
        for l_idx, layer_size in enumerate(dense_layers):
            name = f'dense_{l_idx + 1}'
            dense_layer = tf.keras.layers.Dense(layer_size, activation='relu', name=name)
            dropout_layer = tf.keras.layers.Dropout(dense_drop_rate, name=f'{name}_dropout')
            x = dense_layer(x)
            x = dropout_layer(x)
    output_layer = tf.keras.layers.Dense(num_of_classes, name='output', activation=output_activation)
    output_l = output_layer(x)
    return tf.keras.Model(input_l, output_l)


def get_tcn_model(input_shape: Tuple[int, int],
                  output_activation: str,
                  num_of_classes: int
                  ) -> tf.keras.Model:
    xtr = 6
    model = cnn_tcn(
        nr_features=input_shape[1],
        seq_len=input_shape[0],
        cnn_layers=[
            (16 + xtr, 4, 1),
            (22 + xtr, 3, 2),
            (28 + xtr, 3, 1),
            (34 + xtr, 3, 2),
            (40 + xtr, 3, 1),
            (46 + xtr, 3, 2),
            (52 + xtr, 3, 1),
            (58 + xtr, 3, 2),
            (64 + xtr, 3, 2),
        ],
        # tcn_layers=[],
        tcn_layers=[
            (64 + xtr, 4),
            (64 + xtr, 4),
            (64 + xtr, 4),
            (64 + xtr, 4),
        ],
        tcn_dilation=2,
        spatial_drop_rate=0.05,
        dense_drop_rate=0.2,
        dense_layers=[],
        output_activation=output_activation,
        num_of_classes=num_of_classes,
    )
    return model
