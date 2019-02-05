"""simple_gru_parameters.py

Settings and Data for the simple gru in simple_rnns.py

"""
import numpy as np
from ..dataset_builders.text_data import create_easy_token_extraction_data


batch_size = 16  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 8  # Latent dimensionality of the encoding space.
# num_samples = 10000  # Number of samples to train on.

# Create dataset
print("Creating Data set")
char_string = "abcdefghijklmnopXY"
data_size = 50000
data = create_easy_token_extraction_data(data_size=data_size,
                                         char_string=char_string)

print("Building Vocabulary")
characters = ["START", "STOP", "UNK"] + [ch for ch in char_string]
char_index = {ch: ii for ii, ch in enumerate(characters)}
num_input_characters = len(characters)
num_output_characters = num_input_characters
max_input_length = max([len(txt) for txt in data[:, 0]]) + 2
max_output_length = max([len(token) for token in data[:, 1]]) + 2

print("Building Encodings")
# Placeholders
input_encoding = np.zeros(
            (data_size, max_input_length , num_input_characters),
            dtype='float32')
output_encoding = np.zeros(
            (data_size, max_output_length, num_output_characters),
            dtype='float32')

# These can all be combined into one itterator
# Separated for clarity

## One hot character encoding of input_data
for ii, xx in enumerate(data[:, 0]):
    for jj, ch in enumerate(xx):
        input_encoding[ii, jj, char_index[ch]] = 1

## One hot character encoding of output_data
for ii, xx in enumerate(data[:, 1]):
    for jj, ch in enumerate(xx):
        output_encoding[ii, jj, char_index[ch]] = 1


## TODO: create decoder_input encoding
# this can be set as a constant in the model
# but I am following the keras tutorial
# so I will declare it explicitly
decoder_input_data = np.zeros(
    (data_size, max_output_length, num_output_characters),
    dtype='float32')

## Convert to class Parameters
class Parameters:
    batch_size = 16  # Batch size for training.
    epochs = 10  # Number of epochs to train for.
    latent_dim = 8  # Latent dimensionality of the encoding space.
    # num_samples = 10000  # Number of samples to train on.

    print("Creating Data set")
    char_string = "abcdefghijklmnopXY"
    data_size = 50000
    data = create_easy_token_extraction_data(data_size=data_size,
                                             char_string=char_string)


def do_encoding():
    print("Building Encodings")
    # Placeholders
    input_encoding = np.zeros(
                (data_size, max_input_length , num_input_characters),
                dtype='float32')
    output_encoding = np.zeros(
                (data_size, max_output_length, num_output_characters),
                dtype='float32')

    # These can all be combined into one itterator
    # Separated for clarity

    ## One hot character encoding of input_data
    for ii, xx in enumerate(data[:, 0]):
        for jj, ch in enumerate(xx):
            input_encoding[ii, jj, char_index[ch]] = 1

    ## One hot character encoding of output_data
    for ii, xx in enumerate(data[:, 1]):
        for jj, ch in enumerate(xx):
            output_encoding[ii, jj, char_index[ch]] = 1

class Encoder:
    """class to encode/decode data for a particular rnn

    store vocab for input and output
    convert input strings to onehot encoded input arrays
    convert output strings to onehot encoded output arrays
    and vice versa
    """
    def __init__(self, input_data, output_data,
                 input_vocab=None, output_vocab=None,
                 stop_token=True, unknown_token=True,
                 generate_vocabularies=False):
        """

        """

        pass

    def encode(data, vocabulary):
        """
        input:
            data (array): array of strings to encode
            vocabulary (dict): map of characters to indecies
                EX: {"A":0, "B":1, ...}
        """
        data_size = len(data)
        max_input_length =
        num_input_characters =
        input_encoding = np.zeros(
                    (data_size, max_input_length, num_input_characters),
                    dtype='float32')
        for ii, xx in enumerate(data[:, 0]):
            for jj, ch in enumerate(xx):
                encoding[ii, jj, vocabulary[ch]] = 1
