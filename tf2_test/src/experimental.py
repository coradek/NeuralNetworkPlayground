"""experimental.py

experimental training functions

for importing into a jupyter notebook to test out transformer training

"""
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical


def train_transformer_experimental(model, config, data=None):
    """
    Transformer training code
    """
    log_dir = "../data/logs/"

#     BUFFER_SIZE = config.BUFFER_SIZE
#     BATCH_SIZE = config.BATCH_SIZE
    EPOCHS = 3 # config.EPOCHS
#     MAX_LENGTH = config.MAX_LENGTH

    # Hyperparameters
    num_layers = config.num_layers
    d_model = config.d_model
    dff = config.dff
    num_heads = config.num_heads
    dropout_rate = config.dropout_rate

    print("preparing data ...")
    train_dataset = data

    transformer = model

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

        train_dataset = data.batch_generator()
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


class FakeData:
    def __init__(self, dataset_len=40, batch_size=4):
        assert dataset_len % batch_size == 0, "batch_size must divide evenly into dataset_len"
        L = dataset_len # len dataset
        x_size = (3, 3)
        y_size = 4
        xx_data_range = 300
        yy_data_range = 3
        batch_size = batch_size # must divide evenly into L
        batch_num = L//batch_size

    #     xx = np.random.randint(0, xx_data_range, (L, x_size[0], x_size[1])) #, dtype="float32")
    #     xx[:, :, 0] = 0
    #     yy = np.random.randint(0, yy_data_range, (L, y_size))

        xx = np.random.randint(1, xx_data_range, (L, x_size[0], x_size[1])) #, dtype="float32")
        xx[:, :, 0] = 1
        yy = np.random.randint(1, yy_data_range, (L, y_size))

        def _add_start_stop_tokens(data, data_range, batch_num, batch_size):
            data = data.reshape(batch_num, batch_size, -1)
            temp_start = data_range * np.ones((batch_num, batch_size, 1))
            temp_stop = data_range * np.ones((batch_num, batch_size, 1)) + 1
            output = np.concatenate([temp_start, data, temp_stop], axis=2)
            output = output.astype(dtype="float32")
            return output
    #     temp = xx
        self.xx = _add_start_stop_tokens(xx, xx_data_range, batch_num, batch_size)
        self.yy = _add_start_stop_tokens(yy, yy_data_range, batch_num, batch_size)

    def batch_generator(self):
        for item in zip(self.xx, self.yy):
            # print(temp.dtype)
            # print(item[0].dtype)
            yield item



class MnistData:
    def __init__(self):
        IMG_SHAPE = (28, 28, 1)
        NUM_CLASSES = 10
        img_rows = IMG_SHAPE[0]
        img_cols = IMG_SHAPE[1]

        ## Load Data:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # data = np.load(path_data)
        # (x_train, y_train), (x_test, y_test) = (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])

        # Normalize and Reshape Data:
        # x_train = x_train.astype('float32') / 255.
        # x_test = x_test.astype('float32') / 255.

        # y_train = to_categorical(y_train, NUM_CLASSES)
        # y_test = to_categorical(y_test, NUM_CLASSES)

        self.y_train = y_train
        self.y_test = y_test
        self.x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        self.x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        # input_shape = (img_rows, img_cols, 1)

        print(x_train.shape)
        print(x_test.shape)

        print(y_train.shape)
        print(y_test.shape)
        print()


    def batch_generator(self, batch_size=30):
        xx = self.x_train.reshape(len(x_train)//batch_size, batch_size, -1)
        yy = self.y_train.reshape(len(y_train)//batch_size, batch_size, -1)
        for batch in zip(xx, yy):
            yield batch
