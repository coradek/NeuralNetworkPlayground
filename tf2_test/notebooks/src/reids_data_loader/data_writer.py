import logging
import os

import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(annotations_path,
             output_name=None,
             max_file_size=100000,
             overwrite=True,
             log_step=5000):
    """Convert dataset to TFRecords file format.

    If the total number of instances to write is greater than `max_file_size`,
    then multiple TFRecords files are generated.

    Parameters
    ----------
    annotations_path : str
        Path to the annotation file.
    output_name : str or None, optional (default=None)
        Output path.
    max_file_size : int, optional (default=100000)
        Maximum number of instances/examples for each TFRecords file.
    overwrite : bool, optional (default=True)
        Overwrite existing datasets.
    log_step : int, optional (default=5000)
        Print log messages every N steps.

    """
    def get_outpath(out_dir, part, overwrite, *args):
        """Get full output path."""
        out_fname = get_outfname(part, *args)
        if not overwrite:
            while out_fname in os.listdir(out_dir):
                part += 1
                out_fname = get_outfname(part, *args)
        part += 1
        out_path = os.path.join(out_dir, out_fname)
        return out_path, part

    def get_outfname(part, out_name, nparts, out_ext):
        """Get output filename."""
        return '{}_part{}of{}{}'.format(out_name, part, nparts, out_ext)

    logging.info("Building a dataset from %s.", annotations_path)

    out_name = output_name
    if out_name is None:
        out_name = os.path.splitext(os.path.basename(annotations_path))[0]
    out_ext = '.tfrecords'
    logging.info("Output file: %s", out_name + out_ext)

    out_dir = os.path.join(os.path.dirname(annotations_path), 'tfrecords')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(annotations_path, 'r') as f:
        nrecords = sum(1 for _ in f)
        logging.info("%d pairs to process.", nrecords)

    # Determine number of output files.
    nparts = nrecords // max_file_size
    nparts += 0 if nrecords % max_file_size == 0 else 1
    logging.info("%d output files to generate.", nparts)

    longest_label = ''
    with open(annotations_path, 'r') as f:
        part = 1

        # Get name of initial output file.
        if nrecords <= max_file_size:
            out_path = out_name + out_ext
        else:
            out_path, part = get_outpath(out_dir, part, overwrite,
                                         out_name, nparts, out_ext)
        writer = tf.python_io.TFRecordWriter(out_path)

        for idx, line in enumerate(f):
            if idx % log_step == 0 and idx > 0:
                logging.info("Processed %d pairs.", idx)

            (img_path, label) = line.rstrip('\n').split('\t', 1)

            if not label:
                logging.error("Missing label. Skipping file %s.", img_path)
                continue

            try:
                with open(img_path, 'rb') as img_file:
                    img = img_file.read()
            except Exception as e:
                logging.error("Error reading file %s. Skipping.", img_path)
                continue

            if len(label) > len(longest_label):
                longest_label = label

            # Get instance/example.
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(img),
                        'label': _bytes_feature(label.encode())
                    }))

            # Write instance/example.
            writer.write(example.SerializeToString())

            if idx % max_file_size == 0 and idx > 0:
                writer.close()

                logging.info("Wrote file: %s", out_path)

                # Get name of current output file.
                out_path, part = get_outpath(out_dir, part, overwrite,
                                             out_name, nparts, out_ext)
                writer = tf.python_io.TFRecordWriter(out_path)

        writer.close()
        logging.info("Wrote file: %s", out_path)

    logging.info("Dataset is ready: %d pairs.", idx + 1)
    logging.info("Longest label (%s): %s", len(longest_label), longest_label)
