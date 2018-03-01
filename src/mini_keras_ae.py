from tensorflow.python.keras.layers import Input, Dense, LeakyReLU
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers import Dropout, Flatten, Reshape



class AE_0(object):
    """
    A ridiculously simple AutoEncoder
    One dense layer to a 9 dimensional bottleneck
    with the same in reverse on the other side

    Interestingly simple auto encoders like this amount to a process
    roughly equivalent to PCA
    """
    def __init__(self, input_shape=28*28*1, bottleneck=10):
        self.input_shape = input_shape
        self.bottleneck = bottleneck
        self.encoder = Sequential([
                            Dense(self.bottleneck, input_dim=self.input_shape),
                            LeakyReLU()
                            ])
        self.decoder = Sequential([
                            Dense(self.input_shape, input_dim=self.bottleneck),
                            LeakyReLU()
                            ])
        self.autoencoder = Sequential([self.encoder, self.decoder])
        self.autoencoder.compile(optimizer='adagrad',
                                 loss='mse',
                                )

class AE_conv_1(object):
    """
    An autoencoder with a single convolutional layer
    this model takes much longer to train, but achieves the similar
    resolution with a 10 dimensional bottleneck to what the AE_0 model achieves
    with a 20 dimensional bottleneck.

    Additionally, this version shows a significant reduction in
    the splotchy artifacts typical of purely dense encoders
    """
    def __init__(self, input_shape=(28,28,1), bottleneck=10):
        self.input_shape = input_shape
        self.bottleneck = bottleneck
        self.encoder = Sequential([
                            Conv2D(4, (3,3),
                                   padding='same',
                                   input_shape=self.input_shape),
                            LeakyReLU(),
                            Flatten(),
                            Dense(self.bottleneck),
                            LeakyReLU(),
                            ])

        self.decoder = Sequential([
                            Dense(28*28*9, input_dim=self.bottleneck),
                            LeakyReLU(),
                            Reshape((28,28,9)),
                            Conv2DTranspose(1, (3,3),
                                            padding='same'),
                            LeakyReLU()
                            ])
        self.autoencoder = Sequential([self.encoder, self.decoder])
        self.autoencoder.compile(optimizer='adagrad',
                                 loss='mse',
                                )


class AE_dense_1(object):
    """
    A Dense Autoencoder with an extra layer that has approximately the
    same number of hyper parameters as the AE_conv_1 version above.

    Despite the extra layer and accompanying hyper parameters its performance
    is equivalent to that of the AE_0 model over 5 epochs
    """
    def __init__(self, input_shape=28*28*1, bottleneck=10):
        self.input_shape = input_shape
        self.bottleneck = bottleneck
        self.encoder = Sequential([
                            Dense(128, input_dim=self.input_shape),
                            Dense(self.bottleneck),
                            LeakyReLU()
                            ])
        self.decoder = Sequential([
                            Dense(self.input_shape, input_dim=self.bottleneck),
                            LeakyReLU()
                            ])
        self.autoencoder = Sequential([self.encoder, self.decoder])
        self.autoencoder.compile(optimizer='adagrad',
                                 loss='mse',
                                )
