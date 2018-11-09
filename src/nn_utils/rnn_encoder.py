"""
A simple encoder/decoder for use with sequence2sequence text models

"""


## QUESTION: where/how to handle additional/special
#  vocab tokens/characters (e.g. START, STOP, UNKNOWN etc)
class Encoder:
    def __init__(self,
                 input_source=None,
                 input_vocab=None,
                 output_source=None,
                 output_vocab=None,
                 ):
        pass

    def generate_encodings(self):
        # QUESTION: generate input and output vocab at the
        # same time or separate these?
        pass

    def encode(self, input):
        pass

    def decode(self, input):
        pass

def build_vocabulary(input,
                     to_file=False, #False or path
                     add_start=True,
                     add_stop=True,
                     add_unknown=True,
                     ):
    """Build vocabulary from given Dataset"""
    # what format for input?
    # path to file, list, numpy/pandas, other?
    pass
