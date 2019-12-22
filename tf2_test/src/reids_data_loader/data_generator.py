import json
import logging
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image as PILImage
try:
    from wand.image import Image as WandImage
except BaseException:
    pass

from .data_batcher import DataBatcher
from .data_reader import DataReader

MULTIPART_IN_TYPES = [b'GIF87a', b'GIF89a', b'%PDF']
MULTIPART_OUT_TYPE = 'PNG'

#JPEG_QUALITY = [80, 85, 90, 95, 100]
JPEG_QUALITY = range(0, 101, 1)


class suppress_output(object):
    """A context manager for doing a "deep suppression" of stdout and stderr
    in Python. It will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function. It will not suppress raised exceptions."""

    def __init__(self):
        # Open a pair of null files.
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2).
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors.
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def pdf_to_images(file_bytes, format='PNG'):
    """Extract individual images from multipart files (GIFs, PDFs, etc.).

    Parameters
    ----------
    file_bytes : bytes
        Byte-encoded file.
    format : str, optional (default='PNG')
        Format in which to export images.

    Returns
    -------
    page_images : list
        Pages extracted as images.

    Notes
    -----
    Requires ImageMagick and the Wand Python package.

    """
    with suppress_output():  # suppress conversion warnings
        with WandImage(blob=file_bytes, resolution=200) as img:
            page_images = []
            for page_wand_image_seq in img.sequence:
                page_wand_image = WandImage(page_wand_image_seq)
                page_img_bytes = page_wand_image.make_blob(format)
                page_images.append(page_img_bytes)
            return page_images


def convert_text(text, textmap, eos_token, unk_token, onehot=True):
    """Convert input text to a character map index.

    Parameters
    ----------
    text : Any
        Text.
    textmap : str
        Text map to use as input text vocabulary.
    eos_token : int
        End-of-sentence token index.
    unk_token : int
        Unknown character token index.
    onehot : bool, optional (default=True)
        One-hot encode the text mappings.

    Returns
    -------
    Array of mapped text values. If `onehot=True`, returns the mappings
    one-hot encoded. Otherwise, returns the direct mapping.

    """
    mapping = []
    for char in str(text):
        try:
            mapping.append(textmap.index(char))
        except:
            mapping.append(unk_token)
    mapping += [eos_token]

    if onehot:
        # One-hot encode.
        encoding = np.zeros([len(mapping), len(textmap)], dtype=np.int32)
        encoding[np.arange(len(mapping)), mapping] = 1
        return np.array(encoding, dtype=np.int32)

    return np.array(mapping, dtype=np.int32)


def convert_label(label,
                  labelmap,
                  go_token,
                  eos_token,
                  unk_token,
                  reverse_labels=False,
                  force_uppercase=False):
    """Convert a label to a character map index.

    Parameters
    ----------
    label : Any
        Label.
    labelmap : str
        Label map to use as output label vocabulary.
    go_token : int
        Go token index.
    eos_token : int
        End-of-sentence token index.
    unk_token : int
        Unknown character token index.
    reverse_labels : bool, optional (default=False)
        Reverse all label values.
    force_uppercase : bool, optional (default=False)
        Force labels to uppercase.

    Returns
    -------
    Array of mapped label values.

    """
    if not isinstance(label, (list, np.ndarray)):
        label = str(label)

    if reverse_labels:
        label = label[::-1]
    if force_uppercase:
        label = label.upper()

    mapping = []
    for value in label:
        try:
            mapping.append(labelmap.index(str(value)))
        except:
            mapping.append(unk_token)
    mapping = [go_token] + mapping + [eos_token]

    return np.array(mapping, dtype=np.int32)


class DataGenerator(object):
    """Dataset generator.

    Generates batches of instances.

    Parameters
    ----------
    input_type : str
        Type of input data ('image' or 'text').
    phase : str
        Model phase ('train', 'test', 'predict', or 'extract').
    epochs : int, optional (default=-1)
        Number of epochs to generate data. If -1, generate data indefinitely.
    buffer_size : int, optional (default=10000)
        The buffer size to use for shuffling TFRecords data.
    data_shuffle : bool, optional (default=True)
        Shuffle data.
    max_label_len : int, optional (default=-1)
        Maximum allowable output label length. If -1, then no limit.
    reverse_labels : bool, optional (default=False)
        Reverse all label values.
    force_uppercase : bool, optional (default=False)
        Force labels to uppercase.
    allow_decimals : bool, optional (default=True)
        Allow decimal values in labels.
    skip_multipage : bool, optional (default=False)
        Skip multipart image files (e.g., GIFs, PDFs, etc.).
        Only used if `input_type='image'`.
    max_text_len : int, optional (default=2000)
        Maximum allowable input text length.
        Only used if `input_type='text'`.
    keep_over_max : bool, optional (default=False)
        If True, slice text that exceeds `max_text_len` to the maximum.
        Otherwise, omit those instances from the batches.
        Only used if `input_type='text'`.
    onehot_text : bool, optional (default=True)
        One-hot encode the text mappings.
        Only used if `input_type='text'`.
    bucket_boundaries : list of ints, optional (default=None)
        The edges of the buckets to use when bucketing text data. Two extra
        buckets are created, one for input_length < bucket_boundaries[0] and
        one for input_length >= bucket_boundaries[-1]. If None, bucketing is
        not used.
    input_schema : list or tuple, optional (default=None)
        List or tuple of column names to read as input. Expects `label`. If
        `input_type='image'`, then also expects `filename`. If
        `input_type='text'`, then also expects `text`. Will not use any None
        columns.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    """
    GO_TOKEN = 0
    EOS_TOKEN = 1
    UNK_TOKEN = 2
    LABELMAP = []
    TEXTMAP = []

    def __init__(self,
                 input_type,
                 phase,
                 epochs=-1,
                 buffer_size=10000,
                 data_shuffle=True,
                 max_label_len=-1,
                 reverse_labels=False,
                 force_uppercase=False,
                 allow_decimals=True,
                 skip_multipage=False,
                 max_text_len=2000,
                 keep_over_max=False,
                 onehot_text=True,
                 bucket_boundaries=None,
                 input_schema=None,
                 random_state=None):
        self.input_type = input_type
        self.phase = phase

        # Label params.
        self.max_label_len = max_label_len
        self.reverse_labels = reverse_labels
        self.force_uppercase = force_uppercase
        self.allow_decimals = allow_decimals

        # Input image params (as applicable).
        self.skip_multipage = skip_multipage

        # Input text params (as applicable).
        self.max_text_len = max_text_len
        self.keep_over_max = keep_over_max
        self.onehot_text = onehot_text

        if bucket_boundaries is not None:
            if not isinstance(bucket_boundaries, (list, tuple)):
                raise TypeError("`bucket_boundaries` must be a list or tuple,"
                                " but received %s." % bucket_boundaries)
            for (s, e) in zip(bucket_boundaries[:-1], bucket_boundaries[1:]):
                if not isinstance(s, int) or not isinstance(e, int):
                  raise TypeError("Bucket boundaries must be integers, but"
                                  " saw: %s and %s." % (s, e))
                if s >= e:
                  raise ValueError(
                    "Bucket boundaries must be sequentially increasing"
                    " lengths, but saw: %s before %s." % (s, e))
            buckets_min = [np.iinfo(np.int32).min]
            buckets_max = [np.iinfo(np.int32).max]
            bucket_boundaries = (buckets_min +
                                 list(bucket_boundaries) +
                                 buckets_max)
            self.bucket_boundaries = bucket_boundaries
        else:
            self.bucket_boundaries = [np.iinfo(np.int32).max]

        self.random_state = random_state

        self.is_training = True if phase == 'train' else False

        self.data_reader = DataReader(
            input_type,
            self.is_training,
            epochs=epochs,
            buffer_size=buffer_size,
            data_shuffle=data_shuffle,
            max_text_len=max_text_len,
            keep_over_max=keep_over_max,
            input_schema=input_schema,
            random_state=random_state
        )

        self.reset_batches()

    @staticmethod
    def set_textmap(textmap, is_sequence=False):
        """Set the text map."""
        if is_sequence:
            DataGenerator.TEXTMAP = ['', '', '']

        with open(textmap) as f:
            DataGenerator.TEXTMAP += json.load(f)

    @staticmethod
    def set_labelmap(labelmap,
                     is_sequence=False,
                     full_ascii=False,
                     allow_decimals=True):
        """Set the label map."""
        if is_sequence:
            DataGenerator.LABELMAP = ['', '', '']

        if full_ascii:
            DataGenerator.LABELMAP += [chr(i) for i in range(32, 127)]
        else:
            DataGenerator.LABELMAP += list(labelmap)

        if not allow_decimals:
            try:
                idx = DataGenerator.LABELMAP.index(chr(46))
                del DataGenerator.LABELMAP[idx]
            except Exception as e:
                pass

    def set_dataset(self, data_path):
        """Set a dataset file or directory."""
        self.data_reader.set_dataset(data_path)

    def reset_batches(self):
        """Reset batch data."""
        if self.bucket_boundaries is not None:
            self.data_batcher = {
                i: DataBatcher() for i in range(len(self.bucket_boundaries))
            }
        else:
            self.data_batcher = {0: DataBatcher()}

    def generate(self, batch_size, **kwargs):
        """Generate a batch of instances.

        Parameters
        ----------
        batch_size : int
            Size of minibatches for stochastic optimizers.
        img_file : bytes or None, optional (default=None)
            If prediction phase, used as the input.

        Yields
        ------
        Batch of `batch_size` number of instances.

        Raises
        ------
        IOError
            If predicting without appropriate keywork argument(s).

        """
        if self.phase == 'predict':
            if self.input_type == 'image':
                # Byte-encoded image.
                if 'img_file' in kwargs:
                    img = BytesIO(kwargs['img_file']).getvalue()
                    for batch in self.process_instance(img, '', 1):
                        yield batch
                else:
                    logging.error("No input image file.")
            elif self.input_type == 'text':
                if 'text' in kwargs:
                    # Raw text.
                    txt = kwargs['text']
                    for batch in self.process_instance(txt, '', 1):
                        yield batch
                else:
                    logging.error("No input text.")
            else:
                raise IOError
        else:
            for instance in self.data_reader.read_data(batch_size, **kwargs):
                inp = instance['input']
                lbl = instance['label']
                fname = None
                if self.phase in set(['extract', 'test']):
                    if 'img_file' in instance.keys():
                        fname = instance['img_file']
                for batch in self.process_instance(inp, lbl, batch_size,
                                                   fname=fname):
                    yield batch

        self.reset_batches()

    def process_instance(self, inp, lbl, *args, **kwargs):
        """Process a single instance of image and label.

        Parameters
        ----------
        inp : bytes or str
            Input. If image, as bytes. If text, as string.
        label : Any
            Label.

        Yields
        ------
        Batch of `batch_size` number of instances.

        Raises
        ------
        NotImplementedError
            If the input type is unsupported.

        """
        if self.input_type == 'image':
            if len(inp) == 0:  # ignore if empty file
                #logging.warning("Ignoring empty file.")
                return
            if inp.startswith(tuple(MULTIPART_IN_TYPES)):
                # Try to process multipart image files (e.g., GIFs and PDFs).
                if 'wand' in sys.modules and not self.skip_multipage:
                    try:
                        for img_i in pdf_to_images(
                            inp, format=MULTIPART_OUT_TYPE):
                            # Recurse with the image.
                            for batch in self.process_instance(
                                img_i, lbl, *args, **kwargs):
                                yield batch
                    except Exception as e:
                        #logging.error(e)
                        logging.error("Error reading multipart file.")
                else:
                    #logging.warning("Skipping multipart file.")
                    pass
            else:
                for batch in self.get_batch(inp, lbl, *args, **kwargs):
                    yield batch 
        elif self.input_type == 'text':
            for batch in self.get_batch(inp, lbl, *args, **kwargs):
                yield batch
        else:
            logging.error("Not a valid input type: `%s`.", self.input_type)
            raise NotImplementedError

    def get_batch(self, inp, label, batch_size, *args, **kwargs):
        """Get a batch of input and label pairs.

        Parameters
        ----------
        inp : Any
            Input.
        label : Any
            Label.
        batch_size : int
            Number of instances to return per batch.

        Yields
        ------
        Batch of `batch_size` number of instances.

        """
        if isinstance(label, bytes):
            label = label.decode()

        if self.is_training and not self.allow_decimals:
            try:
                if not float(label).is_integer():
                    logging.warning("Label value `%s` is a decimal number,"
                                    " but `allow_decimals` is False. Instance"
                                    " will be skipped.", str(label))
                    return
            except Exception:
                pass

        if not self.allow_decimals:
            label = label.split('.')[0]

        resave_imgs = True
        if resave_imgs and self.is_training and self.input_type == 'image':
            # Resave some images with randomly selected compression quality.
            if np.random.choice([0, 1]):
                try:
                    with PILImage.open(BytesIO(inp)) as img:
                        img = img.convert('RGB')
                        with BytesIO() as img_file:
                            quality = np.random.choice(JPEG_QUALITY)
                            quality = np.asscalar(quality)
                            img.save(img_file, format='JPEG', quality=quality)
                            inp = img_file.getvalue()
                except OSError as e:
                    #logging.error(e)
                    pass

        inp = self.convert_input(inp)
        lex = self.convert_label(label)

        if self.is_training:
            if set(lex).issubset(
                set((self.GO_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN))):
                logging.warning("No known value mappings for the label `%s`."
                                " Instance will be skipped.", str(label))
                return

            if len(lex) > self.max_label_len:
                logging.warning("Label length is longer than the maximum"
                                " allowed length of %d. Instance will be"
                                " skipped.", self.max_label_len)
                return

        b_idx = 0
        cur_batch_len = 0
        fname = kwargs['fname'] if 'fname' in kwargs else None
        if self.input_type == 'image':
            cur_batch_len = self.data_batcher[b_idx].append(
                lex,
                label,
                fname=fname,
                input_image=inp,
                reverse_labels=self.reverse_labels,
                force_uppercase=self.force_uppercase,
            )
        elif self.input_type == 'text':
            if self.is_training:
                b_idx = next(x[0] for x in enumerate(
                    self.bucket_boundaries) if x[1] >= len(inp))
            cur_batch_len = self.data_batcher[b_idx].append(
                lex,
                label,
                input_text=inp,
                reverse_labels=self.reverse_labels,
                force_uppercase=self.force_uppercase,
            )

        if cur_batch_len >= batch_size:
            equal_inp_len = self.input_type == 'text'
            max_input_len = None
            #if self.input_type == 'text':
            #    max_input_len = self.max_text_len
            batch = self.data_batcher[b_idx].get_batch(
                self.EOS_TOKEN,
                self.max_label_len,
                max_input_len=max_input_len,
                equal_inp_len=equal_inp_len
            )
            yield batch

    def convert_input(self, inp):
        """Convert input."""
        if self.input_type == 'text':
            inp = convert_text(
                inp,
                self.TEXTMAP,
                self.EOS_TOKEN,
                self.UNK_TOKEN,
                onehot=self.onehot_text
            )
        return inp

    def convert_label(self, label):
        """Convert label."""
        lex = convert_label(
            label,
            self.LABELMAP,
            self.GO_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
            reverse_labels=self.reverse_labels,
            force_uppercase=self.force_uppercase
        )
        return lex
