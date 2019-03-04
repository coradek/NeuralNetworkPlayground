"""simple_gru_config.py

Settings and Data for the simple gru in simple_rnns.py

"""
import numpy as np
from ..dataset_builders.text_data import create_easy_token_extraction_data
from ..nn_utils.rnn_encoder import Encoder


class Config:
    def __init__(self):
        print("Creating Data set")
        data_size = 50000
        # char_string = "abcdefghijklmnopqrstuvwxyz XY"
        # data = create_easy_token_extraction_data(data_size=data_size,
        #                                          char_string=char_string,
        #                                          prefix_len_range=(1,3),
        #                                          suffix_len_range=(2,6),
        #                                          token_len_range=(4,16),
        #                                          )

        char_string = "abcdeXY"
        data = create_easy_token_extraction_data(data_size=data_size,
                                                 char_string=char_string,
                                                 prefix_len_range=(2,3),
                                                 suffix_len_range=(2,3),
                                                 token_len_range=(2,3),
                                                 )

        print("Building Vocabulary")
        # characters = ["[STOP]", "[START]", "[UNK]"] \
        #              + [ch for ch in char_string]
        # characters = ["[PAD]", "[STOP]", "[START]", "[UNK]"] \
        #              + [ch for ch in char_string]
        characters = ["[STOP]", "[UNK]"] + [ch for ch in char_string]
        vocabulary_map = {}
        for ii, ch in enumerate(characters):
            vocabulary_map[ch] = ii
        num_input_characters = len(characters)
        num_output_characters = num_input_characters
        print("Building Encoding")
        encoder = Encoder(vocabulary_map)
        input_encoding = encoder.encode(data[:, 0], encode_as="input")
        output_encoding = encoder.encode(data[:, 1], encode_as="output")
        print("Setting Attributes")
        self.batch_size = 32  # Batch size for training.
        self.epochs = 900  # Number of epochs to train for.
        self.latent_dim = 16  # Latent dimensionality of the encoding space.
        self.data = data
        self.encoder = encoder
        self.num_input_characters = num_input_characters
        self.num_output_characters = num_output_characters
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        print("Done")
