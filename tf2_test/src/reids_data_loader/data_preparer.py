import os

import tensorflow as tf
from PIL import Image

from .data_augmenter import *


def resize_image(image,
                 img_channels,
                 pad_height,
                 pad_width,
                 method=tf.image.ResizeMethod.AREA):
    """Resize the image to a maximum height of `self.pad_height` and width of
    `self.pad_width` while maintaining the aspect ratio. Pad the resized image
    with zeros to a fixed size of ``[self.pad_height, self.pad_width]``, with
    the resized image placed at the top-left of the padded bounding box.

    Parameters
    ----------
    image : tensor
        Input image.
    img_channels : int
        Number of image channels to use as model input.
    pad_height : int
        Image height in pixels to use as model input.
    pad_width : int
        Image width in pixels to use as model input.
    method : TF ResizeMethod, optional (default=tf.image.ResizeMethod.AREA)
        Interpolation method.

    Returns
    -------
    Resized image.

    """
    dims = tf.shape(image)

    height_float = tf.cast(pad_height, tf.float64)
    max_width = tf.to_int32(
        tf.ceil(tf.truediv(dims[1], dims[0]) * height_float))
    max_height = tf.to_int32(
        tf.ceil(tf.truediv(pad_width, max_width) * height_float))

    resized = tf.cond(
        tf.greater_equal(pad_width, max_width),
        lambda: tf.cond(
            tf.greater_equal(pad_height, dims[0]),
            lambda: tf.to_float(image),
            lambda: tf.image.resize_images(image, [pad_height, max_width],
                                           method=method, align_corners=True),
        ),
        lambda: tf.image.resize_images(image, [max_height, pad_width],
                                       method=method, align_corners=True)
    )

    padded = tf.image.pad_to_bounding_box(resized, 0, 0, pad_height, pad_width)
    padded.set_shape([pad_height, pad_width, img_channels])

    return padded


def distort_image(image, random_order=False):
    """Distort the image. With some degree of randomness, apply horizontal and
    vertical flipping, stretching/contracting, shearing, rotating, adjustments
    to brightness and contrast, additive Gaussian noise, and masking noise.

    Parameters
    ----------
    image : tensor
        Input image.
    random_order : bool, optional (default=False)
        Randomly order the distortions.

    Returns
    -------
    Distorted image.

    """
    seq = Sequential(
        [  
            #FlipLeftRight(0.01),
            #FlipUpDown(0.01),
            Rotate(0.5, minval=3.14159, maxval=3.14159),
            #Rotate90(0.75, minval=1, maxval=4),
            StretchOrContract(0.75, minval=0.95, maxval=1.05),
            Shear(0.5, intensity=0.01),
            Rotate(0.75, minval=-0.05, maxval=0.05),
            AdjustBrightness(0.5, max_delta=0.075),
            AdjustContrast(0.5, lower=0.95, upper=1.05),
            AddGaussianNoise(0.5, maxval=0.01),
            AddMaskingNoise(0.5, maxval=0.01)
            #Cutout(0.75, minval=0, maxval=100)  # TODO
        ],
        random_order=random_order)
    image_aug = seq.augment_image(image)
    return image_aug


class DataPreparer(object):
    """Data preparer.

    Prepares images and labels for model input.

    Parameters
    ----------
    img_channels : int
        Number of image channels to use as model input.
    data_augment : bool
        Augment training data.
    pad_height : int
        Image height in pixels to use as model input.
    pad_width : int
        Image width in pixels to use as model input.
    method : TF ResizeMethod, optional (default=tf.image.ResizeMethod.AREA)
        Interpolation method.

    """

    def __init__(self,
                 image_channels,
                 image_augment,
                 image_height,
                 image_width,
                 label_depth,
                 image_resize_method=tf.image.ResizeMethod.AREA):
        self.image_channels = image_channels
        self.image_augment = image_augment
        self.image_height = image_height
        self.image_width = image_width
        self.image_resize_method = image_resize_method
        self.label_depth = label_depth

        self.img = tf.placeholder(tf.string, name='image')
        img = tf.cond(
            tf.less(tf.rank(self.img), 1),
            lambda: tf.expand_dims(self.img, 0),
            lambda: self.img
        )
        self.img_prep = tf.map_fn(self._prepare_image, img, dtype=tf.float32)

        self.label = tf.placeholder(tf.int32, name='label')
        label = tf.cond(
            tf.less(tf.rank(self.label), 1),
            lambda: tf.expand_dims(self.label, 0),
            lambda: self.label
        )
        self.label_prep = tf.map_fn(self._prepare_label, label, dtype=tf.int32)

        self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True))

    def _prepare_image(self, image):
        """Prepare the image for model input.

        Parameters
        ----------
        image : tensor
            Input image.

        Returns
        -------
        Prepared image.

        """
        img = tf.image.decode_png(image, channels=self.image_channels)

        if self.image_augment:
            img = distort_image(img)

        img = resize_image(
            img,
            self.image_channels,
            self.image_height,
            self.image_width,
            self.image_resize_method
        )

        return img

    def _prepare_label(self, indices):
        """Prepare the label for model input.

        Parameters
        ----------
        label : tensor
            Input label.

        Returns
        -------
        Prepared label.

        """
        label = tf.one_hot(indices, self.label_depth, dtype=tf.int32)

        return label

    def _visualize_images(self, input_feed, output_dir):
        """Visualize resized/padded images.

        Parameters
        ----------
        input_feed : dict
            Feed dict.
        output_dir : str
            Output directory.

        """
        output_dir = os.path.join(output_dir, 'images')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_feed = self.sess.run([self.img_prep], input_feed)

        for img in img_feed[0]:
            filestring = '1.png'
            idx = 2
            while filestring in os.listdir(output_dir):
                filestring = '{}.png'.format(idx)
                idx += 1
            img = Image.fromarray(np.squeeze(img).astype(np.uint8))
            img.save(os.path.join(output_dir, filestring))

    def prepare_images(self, images, visualize=False, output_dir='out'):
        """Prepare a batch of images.

        Parameters
        ----------
        images : bytes
            Batch of images as bytes.
        visualize : bool, optional (default=False)
            Visualize prepared images.
        output_dir : str, optional (default='out')
            Output directory for visualized images.

        Returns
        -------
        Batch of prepared images.

        """
        input_feed = {}
        input_feed[self.img] = images
        output_feed = [self.img_prep]
        prepared_images = self.sess.run(output_feed, input_feed)[0]

        if visualize:
            self._visualize_images(input_feed, output_dir)

        return prepared_images

    def prepare_labels(self, labels):
        """Prepare a batch of labels.

        Parameters
        ----------
        labels : array of int
            Batch of labels.

        Returns
        -------
        Batch of prepared labels.

        """
        input_feed = {}
        input_feed[self.label] = labels
        output_feed = [self.label_prep]
        prepared_labels = self.sess.run(output_feed, input_feed)[0]

        return prepared_labels
