import logging

import numpy as np


class DataBatcher(object):
    """Data batcher. Handles the batching of instances."""

    def __init__(self):
        self.image_list = []
        self.text_list = []
        self.label_list = []
        self.fname_list = []
        self.ground_list = []

    def append(self,
               label,
               ground,
               fname=None,
               input_image=None,
               input_text=None,
               reverse_labels=False,
               force_uppercase=True):
        """Append an instance to the batch.

        Parameters
        ----------
        input_image : bytes or None, optional (default=None)
            Image as bytes.
        input_text : array of ints or None, optional (default=None)
            Mapped text.
        label : array of ints
            Mapped label.
        ground : Any
            Raw label.
        fname : str
            Filename.
        reverse_labels : bool, optional (default=False)
            Reverse all label values.
        force_uppercase : bool, optional (default=True)
            Force labels to uppercase.

        Returns
        -------
        Length of the updated batch.

        """
        if reverse_labels:
            ground = ground[::-1]
        if force_uppercase:
            ground = ground.upper()

        self.image_list.append(input_image)
        self.text_list.append(input_text)
        self.label_list.append(label)
        self.fname_list.append(fname)
        self.ground_list.append(ground)

        return len(self.label_list)

    def get_batch(self,
                  eos_token,
                  max_label_len,
                  max_input_len=None,
                  equal_inp_len=False):
        """Get a batch of instances.

        Parameters
        ----------
        eos_token : int
            EOS token index.
        max_label_len : int
            Maximum label length. Labels are padded to the label length with
            the EOS token index value. If -1, then no padding is performed.
        max_input_len : int, optional (default=None)
            Maximum input length. Inputs are padded to the input length with
            the EOS token index value. If -1, then no padding is performed.
        equal_inp_len : bool, optional (default=False)
            Equalize the input lengths for the batch. If True, then inputs are
            padded with the EOS token index value to the longest input length.

        Returns
        -------
        Dict of inputs, labels, and grounds for the batch.

        """
        if equal_inp_len:
            if max_input_len is None:
                max_input_len = len(max(self.text_list, key=len))
            for idx in range(len(self.text_list)):
                input_len = len(self.text_list[idx])
                if input_len > max_input_len:
                    self.text_list[idx] = self.text_list[idx][:max_input_len]
                    input_len = len(self.text_list[idx])
                pad_len = max_input_len - input_len
                if len(self.text_list[idx].shape) == 1:
                    # Padding for raw text mappings.
                    pad = np.full([pad_len], eos_token, dtype=np.int32)
                elif len(self.text_list[idx].shape) == 2:
                    # Padding for one-hot encoded text mappings.
                    pad = np.zeros([pad_len, self.text_list[idx].shape[1]],
                                   dtype=np.int32)
                    pad[np.arange(pad_len), eos_token] = 1
                else:
                    raise ValueError(
                        "Text mappings should be 1D or 2D arrays.")
                self.text_list[idx] = np.concatenate(
                    (self.text_list[idx], pad))

        for idx in range(len(self.label_list)):
            label_len = len(self.label_list[idx])
            if label_len <= max_label_len:
                pad_len = max_label_len - label_len
                pad = np.full(pad_len, eos_token, dtype=np.int32)
                self.label_list[idx] = np.concatenate(
                    (self.label_list[idx], pad))
            else:
                logging.warning("Label length is greater than maximum allowed"
                                " length. Label value will be truncated.")
                self.label_list[idx] = np.concatenate(
                    (self.label_list[idx][:(max_label_len - 1)], [eos_token]))

        res = {}
        res['images'] = np.array(self.image_list)
        res['texts'] = np.array(self.text_list)
        res['labels'] = np.array(self.label_list)
        res['fnames'] = self.fname_list
        res['grounds'] = self.ground_list

        self.image_list = []
        self.text_list = []
        self.label_list = []
        self.fname_list = []
        self.ground_list = []

        return res

    def get_batch_len(self):
        return len(self.label_list)

    def __len__(self):
        return len(self.label_list)

    def __iadd__(self, other):
        self.image_list += other.image_list
        self.text_list += other.text_list
        self.label_list += other.label_list
        self.fname_list += other.fname_list
        self.ground_list += other.ground_list

    def __add__(self, other):
        res = DataBucket()
        res.image_list = self.image_list + other.image_list
        res.text_list = self.text_list + other.text_list
        res.label_list = self.label_list + other.label_list
        res.fname_list = self.fname_list + other.fname_list
        res.ground_list = self.ground_list + other.ground_list
        return res
