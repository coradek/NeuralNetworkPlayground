"""
Configuration object for Tensorflow Transformer model
"""

class Config:
    log_dir = "/home/CONCURASP/eadkins/tf2_test/logs"

    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    EPOCHS = 10*9
    MAX_LENGTH = 400   # max input length

    # Hyperparameters
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    # input_vocab_size = tokenizer_pt.vocab_size + 2
    # target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1
