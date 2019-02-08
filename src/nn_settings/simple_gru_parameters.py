"""simple_gru_parameters.py

Settings and Data for the simple gru in simple_rnns.py

"""
import numpy as np
from ..dataset_builders.text_data import create_easy_token_extraction_data



class Parameters:
    def __init__(self):
        print("Creating Data set")
        char_string = "abcdefghijklmnopqrstuvwxyz XY"
        data_size = 50000
        data = create_easy_token_extraction_data(data_size=data_size,
                                                 char_string=char_string,
                                                 prefix_len_range=(1,3),
                                                 suffix_len_range=(2,6),
                                                 token_len_range=(4,16),
                                                 )
        print("Building Vocabulary")
        characters = ["[STOP]", "[START]", "[UNK]"] + [ch for ch in char_string]
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


class Encoder:
    """class to encode/decode data for a particular rnn

    store vocab for input and output
    convert input strings to onehot encoded input arrays
    convert output strings to onehot encoded output arrays
    and vice versa
    """
    def __init__(self,
                 input_vocabulary=None, output_vocabulary=None,
                 max_input_length=None,
                 max_output_length=None,
                 # stop_token=True, unknown_token=True,
                 # add_stop=False,
                 # build_vocabularies=False,
                 ):
        """
        Store input and output vocabularies and maximum input and output lengths
        for a particular dataset/rnn model
        """
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.input_vocabulary = input_vocabulary
        self.reverse_input_vocabulary = {vv: kk for kk, vv
                                         in input_vocabulary.items()}
        if output_vocabulary is not None:
            self.output_vocabulary = output_vocabulary
            self.reverse_output_vocabulary = {vv: kk for kk, vv
                                              in output_vocabulary.items()}
        else:
            self.output_vocabulary = self.input_vocabulary
            self.reverse_output_vocabulary = self.reverse_input_vocabulary

    # def encode_input():
    #     pass
    #
    # def encode_output():
    #     pass
    #
    # def decode_input():
    #     pass
    #
    # def decode_output():
    #     pass

    def encode(self, data, encode_as="input",
               ):
        """
        input:
            data (array): array of strings to encode
            encode_as (str/dict): one of {"input", "output"}
                specifies which vocabulary and what output size
                to use when encoding data
        """
        def get_max_len(data):
            return max([len(txt) for txt in data])

        if encode_as == "input":
            vocabulary = self.input_vocabulary
            if self.max_input_length is None:
                self.max_input_length = get_max_len(data)
            max_length = self.max_input_length
        elif encode_as == "output":
            vocabulary = self.output_vocabulary
            if self.max_output_length is None:
                self.max_output_length = get_max_len(data)
            max_length = self.max_output_length
        ## TODO: Add support for start/stop and UNK characters
        data_size = len(data)
        num_characters = len(vocabulary)
        encoding = np.zeros(
                        (data_size, max_length, num_characters),
                        dtype='float32',
                        )
        ## One hot character encoding of input_data
        for ii, xx in enumerate(data):
            for jj, ch in enumerate(xx):
                encoding[ii, jj, vocabulary[ch]] = 1
        return encoding


    def decode(self, data, decode_as="input"):
        """
        input:
            data (array): array onehot encodings to decode
            vocabulary (str/dict): which vocabulary to use for encoding
                one of {"input", "output"}
                or map of indecies to characters
                EX: {0:"A", 1:"B", ...}
        """
        if decode_as == "input":
            vocabulary = self.reverse_input_vocabulary
            max_length = self.max_input_length
        elif decode_as == "output":
            vocabulary = self.reverse_output_vocabulary
            max_length = self.max_output_length
        output = np.empty(data.shape[0],
                             dtype="S{}".format(max_length)
                             )
        for ii, xx in enumerate(data):
            temp = []
            for jj in xx:
                key = np.argmax(jj)
                ch = vocabulary[key]
                if ch == "[STOP]":
                    break
                temp.append(ch)
            output[ii] = "".join(temp)
        return output

    def build_vocabulary(self, data):
        """
        build vocabulary from given dataset
        (for use when input and/or output vocab is not provided)
        """
        pass
