"""config.py

configuration settings for cnn_rnn
"""
from typing import NamedTuple, Callable, Tuple, Union



class CnnRnnConfig(NamedTuple):
    input_shape: tuple = (None, 28, 56, 1)
    output_size: int = 10       # num possible outputs from rnn (out vocab size)
    max_out_seq_len: int  = 2
    use_gru: bool = True
    latent_dim: int = 16
    use_bidirectional: bool = False
    unroll: bool = False
    cnn_part: int = 1 

class FitConfig(NamedTuple):
    num_epochs: int = 5
    batch_size: int = 32
    validation_split: float = 0.1

class Config(NamedTuple):
    cnn_rnn: NamedTuple = CnnRnnConfig()
    fit: NamedTuple = FitConfig()
