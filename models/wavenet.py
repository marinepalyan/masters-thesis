"""
Source: https://github.com/ultimatist/WaveNet
"""
from typing import Tuple

from keras.layers import Input, Conv1D, Activation, Dropout, Lambda, Multiply, Add, Flatten
from keras.models import Model

# hyper-parameters
n_filters = 32
filter_width = 2
dilation_rates = [2 ** i for i in range(7)] * 2


def get_wavenet_model(input_shape: Tuple[int, int],
                      output_activation: str,
                      num_of_classes: int,
                      ) -> Model:
    # define an input history series and pass it through a stack of dilated causal convolution blocks
    history_seq = Input(shape=input_shape)
    x = history_seq

    skips = []
    for dilation_rate in dilation_rates:
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same', activation='relu')(x)

        # filter
        x_f = Conv1D(filters=n_filters,
                     kernel_size=filter_width,
                     padding='causal',
                     dilation_rate=dilation_rate)(x)

        # gate
        x_g = Conv1D(filters=n_filters,
                     kernel_size=filter_width,
                     padding='causal',
                     dilation_rate=dilation_rate)(x)

        # combine filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])

        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same', activation='relu')(z)

        # residual connection
        x = Add()([x, z])

        # collect skip connections
        skips.append(z)

    # add all skip connection outputs
    out = Activation('relu')(Add()(skips))

    # final time-distributed dense layers
    out = Conv1D(128, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(.2)(out)
    out = Conv1D(num_of_classes, 1, padding='same')(out)
    out = Activation(output_activation)(out)


    # extract training target at end
    def slice(x, seq_length):
        return x[:, -seq_length:, :]


    pred_seq_train = Lambda(slice, arguments={'seq_length': 1})(out)
    pred_seq_train = Flatten()(pred_seq_train)

    model = Model(history_seq, pred_seq_train)
    return model
