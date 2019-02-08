"""simple_rnns.py

A place for basic RNN models

A GRU - Gated Recurrent Unit attempts to solve the vanishing gradient
problem in Recurrent Neural Networks by introducing an Update and Reset Gate
which control how much past information is preserved/forgotten for future steps

"""
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, GRU, Dense
from tensorflow.python.keras.regularizers import l2
# from .nn_settings.simple_gru_parameters import *

## For Mike's amount rnn
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional, RepeatVector, Embedding
from tensorflow.python.keras.optimizers import RMSprop
# from ConcurrenceLayer import Concurrence


def simple_gru(input_size, output_size, latent_dim, max_out_seq_len):
    """
    A simple GRU stripped down about as much as possible
    """
    model = Sequential()
    model.add(GRU(latent_dim,
                  return_sequences=False,
                  input_shape=(None, input_size)
                  ),
              )
    ## RepeatVector repeatedly calls the GRU layer
    ## Storing the output states in an array
    model.add(RepeatVector(max_out_seq_len))
    ## TimeDistributed Dense layer makes a prediction for each
    ## GRU output state in the array from RepeatVector
    model.add(TimeDistributed(Dense(output_size, activation="softmax")
                              )
              )
    rms_prop = RMSprop(
                       rho=0.9,
                       decay=0.0,
                       epsilon=1e-08,
                       )
    model.compile(loss="categorical_crossentropy",
                  optimizer=rms_prop)
    return model


def teacher_forcing_gru_model(num_encoder_tokens,
                              num_decoder_tokens,
                              latent_dim):
    """
    A work in progress
    """
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = GRU(latent_dim, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_gru = GRU(latent_dim, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


def mikes_amount_rnn(input_size, output_size,
                     max_out_seq_len, hidden_size,
                     embedding=False):
    LEARN_RATE = 0.001
    OB_REG = 0.001
    OW_REG = 0.0001
    B_REG = 0.0001
    W_REG = 0.000001
    U_REG = 0.00001
    DROPOUT = 0.10

    model = Sequential()
    if embedding:
        model.add(Embedding(input_size, hidden_size))
        model.add(Bidirectional(GRU(hidden_size,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal',
                        return_sequences=True,
                        kernel_regularizer=l2(W_REG),
                        recurrent_regularizer=l2(U_REG),
                        bias_regularizer=l2(B_REG), dropout=DROPOUT,
                        recurrent_dropout=DROPOUT),
                    merge_mode='concat')
                )
    else:
        model.add(Bidirectional(GRU(hidden_size,
                            kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal',
                            return_sequences=True,
                            kernel_regularizer=l2(W_REG),
                            recurrent_regularizer=l2(U_REG),
                            bias_regularizer=l2(B_REG),
                            dropout=DROPOUT,
                            recurrent_dropout=DROPOUT
                            ),
                        merge_mode='concat', input_shape=(None, input_size)
                        )
                    )
    # model.add(Concurrence())
    model.add(RepeatVector(max_out_seq_len + 1))
    model.add(GRU(hidden_size * 2, kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal', return_sequences=True,
                kernel_regularizer=l2(W_REG), recurrent_regularizer=l2(U_REG),
                bias_regularizer=l2(B_REG), dropout=DROPOUT,
                recurrent_dropout=DROPOUT)
            )
    model.add(TimeDistributed(Dense(output_size, bias_regularizer=l2(OB_REG),
                                    kernel_regularizer=l2(OW_REG), activation="softmax")))
    rms_prop = RMSprop(lr=LEARN_RATE, rho=0.9, decay=0.0, epsilon=1e-08)
    model.compile(loss="categorical_crossentropy", optimizer=rms_prop)
    return model
