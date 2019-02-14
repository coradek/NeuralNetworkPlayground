"""tflite_rnn.py

a place to experiment with rnns built for
converstion to tflite
"""
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, GRU, Dense, LSTM, Lambda
from tensorflow.python.keras.regularizers import l2
# from .nn_settings.simple_gru_config import *

## For Mike's amount rnn
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional, RepeatVector, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from .custom_layers.tflite_rnn_experimental import TFLiteLSTMCell


def gru_functional(input_size, output_size,
                   latent_dim,
                   max_input_length, max_out_seq_len):
    """
    A simple GRU stripped down about as much as possible
    """
    inputs = Input(shape=(max_input_length, input_size,))
    x = GRU(latent_dim,
            return_sequences=False,
            )(inputs)
    ## RepeatVector repeatedly calls the GRU layer
    ## Storing the output states in an array
    x = RepeatVector(max_out_seq_len)(x)
    outputs = TimeDistributed(Dense(output_size, activation="softmax"))(x)
    model = Model(inputs, outputs)
    ## TimeDistributed Dense layer makes a prediction for each
    ## LSTM output state in the array from RepeatVector

    rms_prop = RMSprop(
                       rho=0.9,
                       decay=0.0,
                       epsilon=1e-08,
                       )
    model.compile(loss="categorical_crossentropy",
                  optimizer=rms_prop)
    return model


def lstm_functional(input_size, output_size,
                   latent_dim,
                   max_input_length, max_out_seq_len):
    """
    A simple GRU stripped down about as much as possible
    """
    inputs = Input(shape=(max_input_length, input_size,))
    x = LSTM(latent_dim,
            return_sequences=False,
            )(inputs)
    x = RepeatVector(max_out_seq_len)(x)
    outputs = TimeDistributed(Dense(output_size, activation="softmax")
                              )(x)
    model = Model(inputs, outputs)
    ## TimeDistributed Dense layer makes a prediction for each
    ## GRU output state in the array from RepeatVector
    rms_prop = RMSprop(
                       rho=0.9,
                       decay=0.0,
                       epsilon=1e-08,
                       )
    model.compile(loss="categorical_crossentropy",
                  optimizer=rms_prop)
    return model


def tflite_lstm(input_size, output_size,
                   latent_dim,
                   max_input_length, max_out_seq_len):
    """
    A simple lstm stripped down about as much as possible
    with LSTM cell from tflite experimental
    """
    # def lstm(x):
    #     return TFLiteLSTMCell(x)

    inputs = Input(shape=(max_input_length, input_size,))
    x = TFLiteLSTMCell(latent_dim,
            # return_sequences=False,
            )(inputs)
    # x = Lambda(lstm)(inputs)
    x = RepeatVector(max_out_seq_len)(x)
    outputs = TimeDistributed(Dense(output_size, activation="softmax")
                              )(x)
    model = Model(inputs, outputs)
    ## TimeDistributed Dense layer makes a prediction for each
    ## GRU output state in the array from RepeatVector

    rms_prop = RMSprop(
                       rho=0.9,
                       decay=0.0,
                       epsilon=1e-08,
                       )
    model.compile(loss="categorical_crossentropy",
                  optimizer=rms_prop)
    return model
