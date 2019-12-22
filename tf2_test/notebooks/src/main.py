"""
main entry point for Transformer model

---------------------------------------------------------------------------
From https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb

Copyright 2019 The TensorFlow Authors.
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import os
import time
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

try:
    from config import Config
    from utils import create_masks
    from utils import CustomSchedule
    from utils import LossObject
    from transformer import Transformer

except ModuleNotFoundError:
    #jupyter imports
    print("Using Jupyter Imports")
    from .config import Config
    from .utils import create_masks
    from .utils import CustomSchedule
    from .utils import LossObject
    from .transformer import Transformer




def prepare_data(config,
                 en_encoder_file_path=None,
                 pt_encoder_file_path=None,
                 ):
    MAX_LENGTH = config.MAX_LENGTH
    BUFFER_SIZE = config.BUFFER_SIZE
    BATCH_SIZE = config.BATCH_SIZE

    print("loading dataset")
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   with_info=True,
                                   as_supervised=True,
                                   download=False,
                                   )

    train_examples, val_examples = examples['train'], examples['validation']

    print("building tokenizers")
    if en_encoder_file_path is None:
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
    else:
        tokenizer_en = (tfds.features.text.SubwordTextEncoder
                        .load_from_file(en_encoder_file_path))

    if pt_encoder_file_path is None:
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
    else:
        tokenizer_pt = (tfds.features.text.SubwordTextEncoder
                        .load_from_file(pt_encoder_file_path))

    # ## TEMP:
    # sample_string = 'Transformer is awesome.'
    # tokenized_string = tokenizer_en.encode(sample_string)
    # print("\n")
    # print ('Tokenized string is {}'.format(tokenized_string))
    # print("\n")
    # print("\n")
    # original_string = tokenizer_en.decode(tokenized_string)
    # print ('The original string: {}'.format(original_string))
    # print("\n")
    # assert original_string == sample_string

    def encode(lang1, lang2):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
                 lang1.numpy()) + [tokenizer_pt.vocab_size+1]

        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
                 lang2.numpy()) + [tokenizer_en.vocab_size+1]

        return lang1, lang2

    def tf_encode(pt, en):
        return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

    def filter_max_length(x, y, max_length=MAX_LENGTH):
        """
        filter input length
        """
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    print("filtering and encoding datasets")
    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
                        BATCH_SIZE, padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(tf_encode)
    val_dataset = (val_dataset.filter(filter_max_length)
                             .padded_batch(BATCH_SIZE,
                                           padded_shapes=([-1], [-1]))
                   )

    return train_dataset, val_dataset, tokenizer_en, tokenizer_pt


def train_transformer(config):
    """
    Transformer training code
    """
    log_dir = config.log_dir

    BUFFER_SIZE = config.BUFFER_SIZE
    BATCH_SIZE = config.BATCH_SIZE
    EPOCHS = config.EPOCHS
    MAX_LENGTH = config.MAX_LENGTH

    # Hyperparameters
    num_layers = config.num_layers
    d_model = config.d_model
    dff = config.dff
    num_heads = config.num_heads
    dropout_rate = config.dropout_rate

    print("preparing data ...")
    try:
        (train_dataset, val_dataset,
         tokenizer_en, tokenizer_pt) = prepare_data(config,
                                                    "/home/CONCURASP/eadkins/tf2_test/data/en_tokenizer",
                                                    "/home/CONCURASP/eadkins/tf2_test/data/pt_tokenizer",
                                                    )
    except IOError:
        (train_dataset, val_dataset,
         tokenizer_en, tokenizer_pt) = prepare_data(config)

    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    loss_function = LossObject().loss_function

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        # ckpt.restore(ckpt_manager.latest_checkpoint)
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print ('Latest checkpoint restored!!')

    ## set up logging of Tensorboard events
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, "events", current_time, "train")
    # test_log_dir = log_dir + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        (enc_padding_mask,
         combined_mask,
         dec_padding_mask) = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)


    # def test_step(model, x_test, y_test):
    #     predictions = model(x_test)
    #     loss = loss_object(y_test, predictions)
    #
    #     test_loss(loss)
    #     test_accuracy(y_test, predictions)


    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            ## Tensorboard event logging
            ### QUESTION: should this be less often?
            with train_summary_writer.as_default():
                tf.summary.scalar('loss',
                                  train_loss.result(),
                                  step=epoch)
                tf.summary.scalar('accuracy',
                                  train_accuracy.result(),
                                  step=epoch)

            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                       epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                    train_loss.result(),
                                                    train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    print("We got to here!!!")
    return transformer


if __name__ == '__main__':
    config = Config()
    train_transformer(config)
