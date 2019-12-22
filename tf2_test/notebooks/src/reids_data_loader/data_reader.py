import csv
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    TFRecordDataset = tf.data.TFRecordDataset
except AttributeError:
    TFRecordDataset = tf.contrib.data.TFRecordDataset


class DataReader(object):
    """Dataset reader.

    Reads instances from a data source.

    Parameters
    ----------
    input_type : str
        Type of input data ('image' or 'text').
    is_training : bool
        Training phase.
    epochs : int, optional (default=-1)
        Number of epochs to generate data. If -1, generate data indefinitely.
    buffer_size : int, optional (default=10000)
        The buffer size to use for shuffling TFRecords data.
    data_shuffle : bool, optional (default=True)
        Shuffle data.
    max_text_len : int, optional (default=2000)
        Maximum allowable input text length.
    keep_over_max : bool, optional (default=False)
        If True, slice text that exceeds `max_text_len` to the maximum.
        Otherwise, omit those instances from the buckets.
    input_schema : list or tuple, optional (default=None)
        List or tuple of column names to read as input. Expects `label`. If
        `input_type='image'`, then also expects `filename`. If
        `input_type='text'`, then also expects `text`. Will not use any None
        columns.
    random_state : RandomState instance or None, optional (default=None)
        RandomState instance. If None, the random number generator is the
        RandomState instance used by np.random.

    """
    def __init__(self,
                 input_type,
                 is_training,
                 epochs=-1,
                 buffer_size=10000,
                 data_shuffle=True,
                 max_text_len=2000,
                 keep_over_max=False,
                 input_schema=None,
                 random_state=None):
        self.input_type = input_type
        self.is_training = is_training

        self.epochs = epochs
        self.buffer_size = buffer_size
        self.data_shuffle = data_shuffle

        # Input text params (as applicable).
        self.max_text_len = max_text_len
        self.keep_over_max = keep_over_max

        self.names = input_schema
        if self.names is None:
            self.names = ['filename', 'label', 'text']

        self.usecols = [name for name in set(self.names) if name is not None]

        self.random_state = random_state

        self.records = []

    def set_dataset(self, data_path):
        """Set a dataset file or directory.

        Parameters
        ----------
        data_path : str
            Path to data source. File or directory from which to read input.

        Raises
        ------
        IOError
            If the data path is an empty directory.
        NotImplementedError
            If the data path is not a TFRecords dataset, directory, or file.

        """
        self.data_path = data_path
        self.dataset = None
        if self.data_path is not None:
            if self.data_path.endswith('.tfrecords'):
                self.records = [self.data_path]
                dataset = TFRecordDataset(self.records)
            elif os.path.isdir(self.data_path):
                self.records = []
                for root, dirs, files in os.walk(self.data_path):
                    for file in sorted(files,
                        key=lambda x: os.path.getctime(os.path.join(root, x))):
                        if (file.endswith('.tfrecords')) or (file.endswith('.tfrecord')):
                            self.records.append(os.path.join(root, file))
                if len(self.records) == 0:
                    raise IOError("No training files found.")
                if self.data_shuffle:
                    self.random_state.shuffle(self.records)
                dataset = TFRecordDataset(self.records)
            elif os.path.isfile(self.data_path):
                self.dataset = self.data_path
            else:
                raise NotImplementedError(
                    "Not a valid file or directory: %s." % self.data_path)

            if self.dataset is None:
                dataset = dataset.map(self._parse_image_record)
                if self.data_shuffle:
                    dataset = dataset.shuffle(buffer_size=self.buffer_size)
                self.dataset = dataset.repeat(self.epochs)

    def read_tfrecords(self, batch_size):
        """Iterate over TFRecords.

        Parameters
        ----------
        batch_size : int
            Number of instances to return per batch.

        Yields
        ------
        Dict of instance input and label.

        Raises
        ------
        tf.errors.DataLossError
            If the TFRecords dataset is corrupted or invalid.

        """
        dataset = self.dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        with tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            while True:
                try:
                    if os.path.isdir(self.data_path):
                        records = []
                        for root, dirs, files in os.walk(self.data_path):
                            for file in files:
                                if file.endswith('.tfrecords') \
                                    or file.endswith('.tfrecord') :
                                    records.append(os.path.join(root, file))

                        if set(records) != set(self.records):
                            self.set_dataset(self.data_path)
                            dataset = self.dataset.batch(batch_size)
                            iterator = dataset.make_one_shot_iterator()
                            images, labels = iterator.get_next()
                            logging.info(
                                "Updated training data with %d TFRecords.",
                                len(self.records))
                except Exception as ee:
                    logging.error("Problem checking or updating training data.")
                    raise ee
                    #logging.error(ee)

                try:
                    raw_images, raw_labels = sess.run([images, labels])
                    for img, lbl in zip(raw_images, raw_labels):
                        yield self._to_instance(lbl, img)
                except OSError as e:
                    logging.error(e)
                except (tf.errors.NotFoundError,
                        tf.errors.OutOfRangeError) as e:
                    logging.error(e)
                except tf.errors.DataLossError as e:
                    logging.error(e)
                    ## see if this simplifies error output
                    raise e
                    # raise tf.errors.DataLossError(
                    #     "Invalid TFRecords file(s).")

    def read_tsv(self, batch_size):
        """Iterate over tab-separated values (TSV).

        Parameters
        ----------
        batch_size : int
            Number of instances to return per batch.

        Yields
        ------
        Dict of instance input and label. If image input, also image filename.

        """

        def validate_tsv(filename, ncols):
            """Validates the TSV file is correctly formatted."""
            try:
                data = open(filename).readlines()
                lines = [x.split('\t') for x in data]
                no_newlines = [line for line in lines if len(line) > 1]
                return all(len(line) == ncols for line in no_newlines)
            except Exception as e:
                logging.error(e)
                raise Exception("Error parsing TSV file.")

        def has_header(filename):
            """Determine whether the file has a header."""
            try:
                size = os.path.getsize(filename)
                with open(filename, 'rb') as csvfile:
                    # If there is only one row, then assume no header.
                    for row in csvfile:
                        if len(row) == size:
                            return False
                        break
                    csvfile.seek(0)
                    sniffer = csv.Sniffer()
                    has_header = sniffer.has_header(
                        csvfile.read(2048).decode('utf-8'))
            except Exception as e:
                logging.error(e)
                has_header = False
            return has_header

        epoch_i = 0
        while epoch_i < self.epochs or self.epochs < 0:
            validate_tsv(self.dataset, len(self.names))
            if self.input_type == 'image':
                skiprows = 1 if has_header(self.dataset) else 0
                with open(self.dataset, 'r') as data:
                    rows = data.readlines()[skiprows:]

                    if self.data_shuffle:
                        self.random_state.shuffle(rows)

                    for row in rows:
                        img_file, lbl = row.rstrip('\n').split('\t', 1)
                        if img_file.startswith('./'):
                            # Create full path to images.
                            p = os.path.dirname(os.path.realpath(self.dataset))
                            img_file = os.path.join(p, img_file[2:])
                        try:
                            img = open(img_file, 'rb').read()
                        except (FileNotFoundError, PermissionError) as e:
                            logging.error(e)
                            continue

                        yield self._to_instance(lbl, img, img_file)

            elif self.input_type == 'text':
                dtype = {name: str for name in self.names if name != 'text'}
                converters = {'text': lambda x: x.replace(r'\\r\\n', '\r\n')
                                                 .replace(r'\r\n', '\r\n')}
                skiprows = 1 if has_header(self.dataset) else 0

                chunks = pd.read_csv(self.dataset, sep='\t', header=None,
                                     names=self.names, index_col=False,
                                     usecols=self.usecols, dtype=dtype,
                                     converters=converters, skiprows=skiprows,
                                     chunksize=self.buffer_size, quoting=3,
                                     lineterminator='\n')
                for chunk in chunks:
                    if self.data_shuffle:
                        chunk = chunk.reindex(
                            np.random.permutation(chunk.index))

                    for _, row in chunk.iterrows():
                        txt = str(row['text'])
                        lbl = str(row['label'])

                        if len(txt) > 100000:
                            continue

                        if self.is_training and len(txt) > self.max_text_len:
                            if self.keep_over_max:
                                txt = txt.str.slice(0, self.max_text_len)
                            else:
                                continue

                        yield self._to_instance(lbl, txt)

            epoch_i += 1

    def read_data(self, batch_size, **kwargs):
        """Iterate over data.

        Parameters
        ----------
        batch_size : int
            Number of instances to return per batch.

        Yields
        ------
        Instance dict.

        """
        if (os.path.isdir(self.data_path) or
            self.data_path.endswith('.tfrecords')):
            for instance in self.read_tfrecords(batch_size):
                yield instance
        else:
            for instance in self.read_tsv(batch_size):
                yield instance

    @staticmethod
    def _to_instance(lbl, inp, img_file=None):
        """Create a single instance.

        Parameters
        ----------
        lbl : Any
            Output label.
        inp : Any
            Input.
        img_file : str, optional (default=None)
            Image filename.

        Returns
        -------
        Dict of instance.

        """
        instance = {}
        instance['label'] = lbl
        instance['input'] = inp
        if img_file is not None:
            instance['img_file'] = img_file
        return instance

    @staticmethod
    def _parse_image_record(example_proto):
        """Parse a single image and label pair.

        Parameters
        ----------
        example_proto : tensor
            A single serialized Example.

        Returns
        -------
        Image and label values.

        """
        features = tf.parse_single_example(
            example_proto,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
            })
        return features['image'], features['label']
