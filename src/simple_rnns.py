"""simple_rnns.py

A place for basic RNN models

A GRU - Gated Recurrent Unit attempts to solve the vanishing gradient
problem in Recurrent Neural Networks by introducing an Update and Reset Gate
which control how much past information is preserved/forgotten for future steps

"""
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, GRU, Dense
from tensorflow.python.keras.regularizers import l2
# from keras.models import Model, Sequential
# from keras.layers import Input, GRU, Dense
# from keras.regularizers import l2
# from .nn_settings.simple_gru_parameters import *


def lazy_gru_model(input_size, latent_dim):
    LEARN_RATE = 0.001
    OB_REG = 0.001
    OW_REG = 0.00001
    B_REG = 0.0001
    U_REG = 0.00001
    W_REG = 0.000001
    model = Sequential()
    model.add(GRU(latent_dim, init='glorot_uniform',
                  inner_init='orthogonal',
                  return_sequences=True,
                  U_regularizer=l2(U_REG),
                  W_regularizer=l2(W_REG),
                  b_regularizer=l2(l=B_REG), unroll=False),
                  merge_mode='concat',
                  input_shape=(None, input_size))
    rms_prop = RMSprop(lr=LEARN_RATE,
                       rho=0.9, decay=0.0,
                       epsilon=1e-08)
    model.compile(loss="mean_squared_error",
                  optimizer=rms_prop)

    return model


def simple_gru_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
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
