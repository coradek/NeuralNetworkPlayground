"""data_management.py

get datasets for the cnn_rnn project
"""
import numpy as np
from keras.datasets import mnist


class CnnRnnData:
    def __init__(self, padding_size=(128, 128)):
        self.padding_size = padding_size

    @property
    def pair_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        (x_train, y_train) = make_pair_data(x_train, y_train)
        (x_test, y_test) = make_pair_data(x_test, y_test)
        return (x_train, y_train), (x_test, y_test)

    @property
    def noise_padded_data(self):
        data = self.pair_data
        xtrain, xtest = data[0][0], data[1][0]
        xtrain = make_noise_padded_images(xtrain)
        xtest = make_noise_padded_images(xtest)
        return (xtrain, data[0][1]), (xtest, data[1][1])


def make_pair_data(xx, yy):
    """
    horitontally stitch two input images arrays together
    and provide appropriate answer pair
    for an entire dataset
    """
    assert len(xx) == len(yy)
    order_a = np.random.permutation(len(xx))
    order_b = np.random.permutation(len(xx))
    xa = xx[order_a]
    xb = xx[order_b]
    ya = yy[order_a]
    yb = yy[order_b]
    x_out = np.dstack((xa, xb))
    y_out = np.dstack((ya, yb)).reshape(len(yy), 2)
    return x_out, y_out


def make_n_ple_data(xx, yy,
                    len_range=(1,4)
                    ):
    """
    horitontally stitch N input image arrays together
    and provide appropriate answer sequences
    for an entire dataset
    """
    # assert len(xx) == len(yy)
    # order_a = np.random.permutation(len(xx))
    # order_b = np.random.permutation(len(xx))
    # xa = xx[order_a]
    # xb = xx[order_b]
    # ya = yy[order_a]
    # yb = yy[order_b]
    # x_out = np.dstack((xa, xb))
    # y_out = np.dstack((ya, yb)).reshape(len(yy), 2)
    # return x_out, y_out
    pass


def make_noise_padded_images(data, target_size=(128, 128)):
    """
    pad images with noise so that the original image is located randomly
    """
    data_len = data.shape[0]
    shape = data.shape[1:]
    msg = "{} image will not fit in {} image"
    assert target_size >= shape, msg.format(shape, target_size)
    data_out = np.random.randint(0, 255,
                                 (data_len, target_size[0], target_size[1])
                                )
    for ii in range(data_len):
        xx = np.random.randint(0, 1 + target_size[0] - shape[0])
        yy = np.random.randint(0, 1 + target_size[1] - shape[1])
        data_out[ii, xx : xx+shape[0], yy : yy+shape[1]] = data[ii]
    return data_out
