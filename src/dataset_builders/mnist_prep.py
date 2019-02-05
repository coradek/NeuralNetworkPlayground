"""mnist_prep.py

prepare mnist data for use with simple cnn
"""
import numpy as np
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
from tensorflow.keras.datasets import mnist


class Mnist:
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
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        print(x_train.shape)
        print(x_test.shape)

        y_train = to_categorical(y_train, NUM_CLASSES)
        y_test = to_categorical(y_test, NUM_CLASSES)

        print(y_train.shape)
        print(y_test.shape)
        print()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        self.IMG_SHAPE = IMG_SHAPE
        self.NUM_CLASSES = NUM_CLASSES
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        print("final shape (x_train):", x_train.shape)
        print("final shape (x_test):", x_test.shape)
        print("final shape (y_train):", y_train.shape)
        print("final shape (y_test):", y_test.shape)
        print()
