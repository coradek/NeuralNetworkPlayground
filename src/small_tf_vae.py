"""
A smallish Variational AutoEncoder in Tensorflow
Thank you Felix Mohr
Code From
https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776

A Variational AutoEncoder uses an encoder neural network
to map input values to a smaller dimensional space
and then applies a decoder neural network to the resulting
vector to recreate the original output

This removes the need for labeled data as the "label"
is the same as the input

Once trained the decoder is dropped and what remains is a
neural network capable of creating a compact representation
of the data which preserves in original information.

"""
import tensorflow as tf
import numpy as np

# Activation Function
def lrelu(x, alpha=0.3):
    """leaky relu activation
    like relu - linear above 0
    unlike relu - linear (but with very small slope) below 0
    attempts to solve the dying ReLU problem
    https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks
    """
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X_in, keep_prob, n_latent):
    """convert image to representation vector
    """
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2,
                             padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2,
                             padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1,
                             padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)

        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z  = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def decoder(sampled_z, keep_prob, reshaped_dim, inputs_decoder):
    """convert representation vector to image
    """
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img
