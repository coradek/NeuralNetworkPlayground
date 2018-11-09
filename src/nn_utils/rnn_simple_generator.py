"""
A data generator takes in a raw data file (in this case text as a tsv/csv)
and yeilds batches prepared for
In Keras the benefit of using a generator
"""
import pandas as pd
import numpy as np
import re

# MAX_DIGITS = 50
# DATA_LIMIT = 150000
# RECEIPT_LIMIT must not be larger than DATA_LIMIT
# RECEIPT_LIMIT = 2000

MAX_INPUT_LEN = 30000 #cut off text after this many characters
MAX_OUTPUT_LEN = 80 # cut off output after this many characters
## QUESTION: Why does mike have a separate data_limit and RECEIPT_LIMIT?

def _bucket_to_fit(lengths, data_limit, batch_limit):
    """
    bucket data by length to reduce padding
        sort by len
        divide into batches
        pad all items in batch to len of longest (last) item
        yeild prepared batch
    """
    count = 0
    index = 0
    start_index = 0
    blobs = []
    for element in lengths:
        count += 1
        if count * (element[1] + 1) > data_limit or count > batch_limit:
            blobs += [(start_index, index)]
            count = 1
            start_index = index
        index += 1
    blobs += [(start_index, len(lengths))]
    return blobs


def generate_data(filename, input_vocabulary, output_vocabulary,
             sep="\t"
             max_batch_size=32,
             continuous=True,
             max_input_len=MAX_INPUT_LEN):
    """
    :param: filename - name of file containing pandas readable, utf-8 encoded
            text and label (without header or index)
    :param: input_vocabulary - list of accepted input characters
        or path to file containing line-separated characters
    :param: output_vocabulary - list of accepted input characters
        or path to file containing line-separated characters
    """
    keep_going = True
    while keep_going:
        keep_going = continuous
        chunks = pd.read_csv(filename, sep=sep,
                             index_col=None, header=None,
                             names=("input", "output"),
                             encoding='utf-8', chunksize=10000)
        for chunk in chunks:
            lengths = sorted([(i, len(str(x))) for (i, x)
                              in zip(chunk.index, chunk.input)
                              if  (len(str(x)) < max_input_len)
                                & (isinstance(x, str))], key=lambda x: x[1])
            blobs = _bucket_to_fit(lengths, MAX_INPUT_LEN, max_batch_size)
            np.random.shuffle(blobs)

            for blob in blobs:
                bucket_length = lengths[blob[1] - 1][1] + 1
                indices = [x[0] for x in lengths[blob[0]:blob[1]]]
                bucket = chunk.ix[indices]
                features = np.zeros([len(bucket), bucket_length, len(rcpt_map)], dtype=np.float32)
                labels = np.zeros([len(bucket), bucket_length], dtype=np.float32)
                i = 0
                for _, example in bucket.iterrows():
                    encoder_input = example.input
                    for j in range(len(encoder_input)):
                        if encoder_input[j] in rcpt_map:
                            features[i, j, rcpt_map[encoder_input[j]]] = 1
                        else:
                            features[i, j, rcpt_map['UNK']] = 1
                    features[i, len(encoder_input), rcpt_map['STOP']] = 1
                    for j in range(len(encoder_input) + 1, bucket_length):
                        features[i, j, rcpt_map['PAD']] = 1
                    decoder_output = example.output
                    output_location = example.input.rfind(decoder_output)
                    for j in range(len(decoder_output)):
                        labels[i, j + output_location] = 1
                    i += 1

                yield features, labels
