"""
Simple Convolutional Neural Networks
works well with MNIST
"""
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, LeakyReLU
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.python.keras.layers import Dropout, Flatten, Reshape

class SimpleCNN:
    """takes dims parameter for experimentation"""
    def __init__(self, input_shape=(28,28,1), num_classes=10, dims=[32, 64, 128]):
        model = Sequential()
        model.add(Conv2D(dims[0],
                         kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(dims[1], (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(dims[2], activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss="categorical_crossentropy",
                      optimizer="adagrad",
                      metrics=['accuracy'])
        self.cnn=model

def cnn_adi_1(image_width,image_height):
    weight_decay=1e-4
    #model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',data_format='channels_last',input_shape=(image_width,image_height,3)))
    #model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    #model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    #model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    #model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    #model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    #model.add(Conv2D(128, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(2,2))
    #model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model
