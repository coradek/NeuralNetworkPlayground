"""
train the simple gru on create_easy_token_extraction_data
"""
import os
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
from src.nn_settings.simple_gru_config import Config, Encoder
from src.simple_rnns import simple_gru, mikes_amount_rnn

def main():
    config = Config()
    batch_size = config.batch_size
    epochs = config.epochs
    latent_dim = config.latent_dim
    data = config.data
    num_input_characters = config.num_input_characters
    num_output_characters = config.num_output_characters
    input_encoding = config.input_encoding
    output_encoding = config.output_encoding
    enc = config.encoder
    max_output_length = enc.max_output_length

    print(num_input_characters)
    print(len(data))
    print(data[:5])

    save_path = "../data/keras_rnn.h5"
    if not os.path.isfile(save_path):
        model = simple_gru(num_input_characters,
                   num_output_characters,
                   latent_dim,
                   max_output_length,
                  )
    else:
        model = load_model(save_path)

    # define the checkpoint
    checkpoint = ModelCheckpoint(save_path,
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]

    # fit the model
    model.fit(input_encoding, output_encoding,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1)



if __name__ == '__main__':
    main()
