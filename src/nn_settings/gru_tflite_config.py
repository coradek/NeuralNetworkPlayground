"""gru_tflite_config.py

Config File for gru_for_tflite

"""
import inspect
from typing import NamedTuple, Union, Callable
from ..dataset_builders.text_data import create_easy_token_extraction_data
from ..nn_utils.rnn_encoder import Encoder


def build_vocab(voc_input, build_from="string"):
    # characters: list = ["[STOP]", "[START]", "[UNK]"]
    # characters: list = ["[PAD]", "[STOP]", "[START]", "[UNK]"]
    characters: list = ["[STOP]", "[UNK]"]
    if build_from == "string":
        characters += [ch for ch in voc_input]
        vocabulary: dict = {}
        for ii, ch in enumerate(characters):
            vocabulary[ch] = ii
    else:
        print("build_from must be one of ['string']")
    return vocabulary



class DataConfig(NamedTuple):
    data_size: int = 30000
    char_string: str = "abcdeXY"
    prefix_len_range: tuple = (2,3)
    suffix_len_range: tuple = (2,3)
    token_len_range: tuple = (2,3)



# class Config(NamedTuple):
#     batch_size = 32  # Batch size for training.
#     epochs = 3  # Number of epochs to train for.
#     latent_dim = 16  # Latent dimensionality of the encoding space.
#
#     print("Creating Data set")
#     data_config = DataConfig()
#     data = create_easy_token_extraction_data(**data_config._asdict())
#
#     print("Building Vocabulary")
#     vocabulary_map: dict = build_vocab(data_config.char_string)
#     num_input_characters: int = len(vocabulary_map)
#     num_output_characters: int = num_input_characters
#
#     print("Building Encoding")
#     encoder = Encoder(vocabulary_map)
#     input_encoding = encoder.encode(data[:, 0], encode_as="input")
#     output_encoding = encoder.encode(data[:, 1], encode_as="output")
#     max_input_length = encoder.max_input_length
#     max_output_length = encoder.max_output_length


class Config():
    def __init__(self):
        batch_size = 32  # Batch size for training.
        epochs = 4  # Number of epochs to train for.
        latent_dim = 8  # Latent dimensionality of the encoding space.

        print("Creating Data set")
        data_config = DataConfig()
        data = create_easy_token_extraction_data(**data_config._asdict())

        print("Building Vocabulary")
        vocabulary_map: dict = build_vocab(data_config.char_string)
        num_input_characters: int = len(vocabulary_map)
        num_output_characters: int = num_input_characters

        print("Building Encoding")
        encoder = Encoder(vocabulary_map)
        input_encoding = encoder.encode(data[:, 0], encode_as="input")
        output_encoding = encoder.encode(data[:, 1], encode_as="output")
        max_input_length = encoder.max_input_length
        max_output_length = encoder.max_output_length

        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.data_config = data_config
        self.data = data
        self.vocabulary_map = vocabulary_map
        self.num_input_characters = num_input_characters
        self.num_output_characters = num_output_characters
        self.encoder = encoder
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
