"""cnn_rnn.py

Neural Networks for cnn-rnn project
"""
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers import (Activation, Bidirectional, Dense, Input, Lambda,
                          RepeatVector, Reshape, TimeDistributed)
from keras.layers.recurrent import GRU, LSTM
from keras.optimizers import RMSprop


def simple_cnn(inp):
    """
    the cnn part of a cnn-rnn model
    """
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inp)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)

    # x = Reshape((1, -1))(x)    # Cannot be Unrolled
    temp = x._keras_shape
    temp = temp[1] * temp[2]
    x = Reshape((temp, -1))(x)
    cnn = x

    return cnn


def simple_rnn(inputs,
               output_size,
               max_out_seq_len,
               latent_dim=16,
               use_gru=False,
               use_bidirectional=False,
               unroll=False,
              ):
    """
    A simple LSTM/GRU stripped down about as much as possible
    (uncompiled) for use with cnn-rnn
    """
    RNN = GRU if use_gru else LSTM
    # inputs = Input(shape=(max_input_length, input_size,))
    if use_bidirectional:
        x = Bidirectional(
                RNN(latent_dim,
                    return_sequences=False,
                    unroll=unroll,
                ),
                merge_mode='concat',
            )(inputs)
        latent_dim_2 = latent_dim * 2
    else:
        x = RNN(latent_dim,
                return_sequences=False,
                unroll=unroll,
               )(inputs)
        latent_dim_2 = latent_dim
    x = RepeatVector(max_out_seq_len)(x)
    x = RNN(latent_dim_2,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            return_sequences=True,
            unroll=unroll,
           )(x)
    outputs = TimeDistributed(
                        Dense(output_size, activation="softmax")
                    )(x)

    return outputs


def simple_cnn_2(inp):
    """
    Experimental cnn part of a cnn-rnn model

    Notes:
    reducing
      x = Conv2D(16, (3, 3), [strides=(2, 4)], ...
    to
      x = Conv2D(16, (4, 16), strides=(2, 4), ...

    gives 4x speed-up (20 min -> 5 min) with no apparent performance loss
    (over 3 epochs) on 128x128 noise padded data
    [val_acc: 0.3637]
    """
    x = Conv2D(16, (4, 16), strides=(2, 4),
               padding='same', activation='relu')(inp)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(16, (4, 16), strides=(2, 4),
               padding='same', activation='relu')(inp)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)

    # x = Reshape((1, -1))(x)    # Cannot be Unrolled
    temp = x._keras_shape
    temp = temp[1] * temp[2]
    x = Reshape((temp, -1))(x)
    cnn = x

    return cnn


def simple_cnn_3(inp):
    """
    Experimental cnn part of a cnn-rnn model

    Notes:
    Further reduction from cnn_2

    [trial 1]
    16, (8, 32), strides=(4, 16)
    16, (8, 32), strides=(4, 16)
    32, (3, 3) [remove maxpool]
    137s 3ms/step - val_loss: 1.2506 - val_acc: 0.5394


    [trial 2]
    16, (32, 8), strides=(16, 4)
    16, (32, 8), strides=(16, 4)
    32, (3, 3) [remove maxpool]
    140s 3ms/step - val_loss: 1.2763 - val_acc: 0.5565

    [trial 3]
    16, (32, 32), strides=(16, 16)
    16, (32, 32), strides=(16, 16)
    32, (3, 3) [remove maxpool]
    56s 1ms/step - val_loss: 1.9990 - val_acc: 0.2641
    ** not learning as much per epoch, but epochs take half as long
    ** 6 epochs - still not as good
    59s 1ms/step - val_loss: 1.8661 - val_acc: 0.3071

    [trial 4]
    16, (16, 16), strides=(8, 8)
    16, (16, 16), strides=(8, 8)
    32, (3, 3) [remove maxpool]
    136s 3ms/step - val_loss: 1.1562 - val_acc: 0.5892

    [trial 5] reshape_1 (None, 32, 32)
    16, (8, 8), strides=(4, 4)
    16, (8, 8), strides=(4, 4)
    32, (3, 3) [maxpool INCLUDED]
    """
    x = Conv2D(16, (8, 8), strides=(4, 4),
               padding='same', activation='relu')(inp)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(16, (8, 8), strides=(4, 4),
               padding='same', activation='relu')(inp)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)

    # x = Reshape((1, -1))(x)    # Cannot be Unrolled
    temp = x._keras_shape
    temp = temp[1] * temp[2]
    x = Reshape((temp, -1))(x)
    cnn = x

    return cnn


def cnn_rnn(input_shape=(None, 28, 56, 1),
            output_size=10,
            max_out_seq_len=2,
            latent_dim=16,
            use_gru=False,
            use_bidirectional=False,
            unroll=True,
            cnn_part=1,
            ):
    """
    A fully assembled and compiled cnn-rnn model
    """
    inp = Input(batch_shape=(input_shape))
    if cnn_part == 1:
        cnn = simple_cnn(inp)
    elif cnn_part == 2:
        cnn = simple_cnn_2(inp)
    elif cnn_part == 3:
        cnn = simple_cnn_3(inp)
    else:
        msg = "no corresponding cnn found: {}".format(cnn_part)
        raise NotImplementedError(msg)
    rnn = simple_rnn(cnn,
               output_size,
               max_out_seq_len,
               latent_dim,
               use_gru,
               use_bidirectional,
               unroll,
               )
    model = Model(inp, rnn)
    rms_prop = RMSprop(
               rho=0.9,
               decay=0.0,
               epsilon=1e-08,
               )
    model.compile(loss="categorical_crossentropy",
                  optimizer=rms_prop,
                  metrics=['accuracy'],
                 )
    return model
