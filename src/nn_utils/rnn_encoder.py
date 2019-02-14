"""
A simple encoder/decoder for use with sequence2sequence text models

"""
import numpy as np


def build_vocabulary(v_source, build_from="string"):
    # characters: list = ["[STOP]", "[START]", "[UNK]"]
    # characters: list = ["[PAD]", "[STOP]", "[START]", "[UNK]"]
    characters: list = ["[STOP]", "[UNK]"]
    if build_from == "string":
        characters += [ch for ch in v_source]
        vocabulary: dict = {}
        for ii, ch in enumerate(characters):
            vocabulary[ch] = ii
    else:
        raise ValueError("build_from must be one of ['string']")
    return vocabulary



class Encoder:
    """class to encode/decode data for a particular rnn

    store vocab for input and output
    convert input strings to onehot encoded input arrays
    convert output strings to onehot encoded output arrays
    and vice versa
    """
    def __init__(self,
                 input_vocabulary=None,
                 output_vocabulary=None,
                 max_input_length=None,
                 max_output_length=None,
                 # stop_token=True, unknown_token=True,
                 # add_stop=False, add_unk=False,
                 # build_vocab_from=False,
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
