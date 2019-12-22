import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops


class Augmenter(object):
    """Base class for Augmenter objects.

    All augmenters derive from this class."""

    def __init__():
        pass

    def sometimes(self, prob, func_true, func_false):
        """Apply `func_true` with probability `prob` and `func_false` with
        probability 1 - `prob`."""
        ret = tf.cond(
            tf.less_equal(tf.random_uniform([1])[0], prob),
            lambda: func_true,
            lambda: func_false)
        return ret


class Sequential(Augmenter, list):
    """List augmenter that may contain other augmenters to apply in sequence
    or in random order.

    Raises
    ------
    TypeError
        If `children` is not of type Augmenter, list, or None.

    """

    def __init__(self, children=None, random_order=False):
        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Augmenter):
            list.__init__(self, [children])
        elif isinstance(children, list):
            list.__init__(self, children)
        else:
            raise TypeError(
                "Child augmentations must be a list, Augmenter, or None.")
        self.random_order = random_order

    def augment_image(self, img, random_state=None):
        img = tf.cast(img, dtype=tf.float32)
        if self.random_order:
            if random_state is None or random_state is np.random:
                self.random_state = np.random.mtrand._rand
            elif isinstance(random_state, (numbers.Integral, np.integer)):
                self.random_state = np.random.RandomState(random_state)
            else:
                self.random_state = np.random.mtrand._rand
            for index in random_state.permutation(len(self)):
                img = self[index]._augment_image(img=img)
        else:
            for augmenter in self:
                img = augmenter._augment_image(img=img)
        return img

    def add(self, augmenter):
        """Add an augmenter to the list of child augmenters.

        Parameters
        ----------
        augmenter : Augmenter
            The augmenter to add.

        """
        self.append(augmenter)

    def get_children_lists(self):
        return [self]


class AddGaussianNoise(Augmenter):
    """Add Gaussian noise to the image."""

    def __init__(self, p, minval=0.0, maxval=1.0):
        self.p = p
        self.stddev = tf.random_uniform([1], minval=minval, maxval=maxval)

    def _augment_image(self, img):
        func = img + tf.random_normal(shape=tf.shape(img), stddev=self.stddev)
        img = self.sometimes(self.p, func, img)
        return img


class AddMaskingNoise(Augmenter):
    """Add masking noise to the image."""

    def __init__(self, p, minval=0.0, maxval=0.5):
        self.p = p
        self.masking_prob = tf.random_uniform(
            [1], minval=minval, maxval=maxval)[0]

    def _augment_image(self, img):
        func = tf.nn.dropout(
            img, keep_prob=1 - self.masking_prob) * (1 - self.masking_prob)
        img = self.sometimes(self.p, func, img)
        return img


class AdjustBrightness(Augmenter):
    """Adjust the image brightness."""

    def __init__(self, p, max_delta=0.05):
        self.p = p
        self.max_delta = max_delta

    def _augment_image(self, img):
        func = tf.image.random_brightness(img, max_delta=self.max_delta)
        img = self.sometimes(self.p, func, img)
        return img


class AdjustContrast(Augmenter):
    """Adjust the image contrast."""

    def __init__(self, p, lower=0.9, upper=1.1):
        self.p = p
        self.lower = lower
        self.upper = upper

    def _augment_image(self, img):
        func = tf.image.random_contrast(
            img, lower=self.lower, upper=self.upper)
        img = self.sometimes(self.p, func, img)
        return img


class Cutout(Augmenter):
    """Randomly mask out square regions of the image. TODO."""

    def __init__(self, p, minval=0, maxval=100):
        self.p = p
        self.minval = minval
        self.maxval = maxval

    @staticmethod
    def _create_mask(img, minval=0, maxval=100):
        dims = tf.shape(img)
        cut_size_h = tf.random_uniform(
            [1], minval=minval, maxval=maxval, dtype=tf.int32)[0]
        cut_size_w = tf.random_uniform(
            [1], minval=minval, maxval=maxval, dtype=tf.int32)[0]
        max_cut_h = dims[1] - cut_size_h
        max_cut_w = dims[0] - cut_size_w
        cut_offset_y = tf.random_uniform(
            [1], minval=0, maxval=max_cut_h, dtype=tf.int32)[0]
        cut_offset_x = tf.random_uniform(
            [1], minval=0, maxval=max_cut_w, dtype=tf.int32)[0]
        cut_h = tf.range(cut_offset_y, cut_size_h + cut_offset_y)
        cut_w = tf.range(cut_offset_x, cut_size_w + cut_offset_x)
        i1, i2 = tf.meshgrid(cut_h, cut_w, indexing='ij')
        ref = tf.ones_like(tf.shape(img), dtype=tf.float32)
        indices = tf.stack(
            [tf.reshape(i2, [-1]), tf.reshape(i1, [-1])], axis=-1)
        updates = tf.zeros(tf.shape(indices)[0], dtype=tf.float32)
        updates = tf.expand_dims(tf.reshape(updates, [-1]), -1)
        # `ref` raises AttributeError: 'Tensor' object has no attribute 'handle'.
        mask = tf.scatter_nd_update(ref, indices, updates)
        return mask

    @staticmethod
    def _mask_image(img, mask):
        masked_img = mask * img
        invert_mask = mask * -1.0 + 1.0
        masked_img += mean_pixel * invert_mask
        return masked_img

    def _augment_image(self, img):
        mask = self._create_mask(img, self.minval, self.maxval)
        func = self._mask_image(img, mask)
        img = self.sometimes(self.p, func, img)
        return img


class FlipLeftRight(Augmenter):
    """Randomly flip the image horizontally (left to right)."""

    def __init__(self, p):
        self.p = p

    def _augment_image(self, img):
        func = tf.image.random_flip_left_right(img)
        img = self.sometimes(self.p, func, img)
        return img


class FlipUpDown(Augmenter):
    """Randomly flip the image vertically (upside down)."""

    def __init__(self, p):
        self.p = p

    def _augment_image(self, img):
        func = tf.image.random_flip_up_down(img)
        img = self.sometimes(self.p, func, img)
        return img


class Rotate(Augmenter):
    """Rotate the image by an arbitrary angle."""

    def __init__(self, p, minval=-0.1, maxval=0.1):
        self.p = p
        self.minval = minval
        self.maxval = maxval

    def _augment_image(self, img):
        angles = tf.random_uniform(
            [1], minval=self.minval, maxval=self.maxval)
        func = tf.contrib.image.rotate(img, angles=angles)
        img = self.sometimes(self.p, func, img)
        return img


class Rotate90(Augmenter):
    """Rotate the image counter-clockwise by 90 degrees."""

    def __init__(self, p, minval=0, maxval=4):
        self.p = p
        self.minval = minval
        self.maxval = maxval

    def _augment_image(self, img):
        k = tf.random_uniform(
            [1], minval=self.minval, maxval=self.maxval, dtype=tf.int32)[0]
        func = tf.image.rot90(img, k=k)
        img = self.sometimes(self.p, func, img)
        return img


class Shear(Augmenter):
    """Spatially shear the image."""

    def __init__(self, p, intensity=0.05):
        self.p = p
        shear = tf.random_uniform([1], minval=-intensity, maxval=intensity)[0]
        self.shear_matrix = [[1, -tf.sin(shear), 0],
                             [0, tf.cos(shear), 0],
                             [0, 0, 1]]

    @staticmethod
    def _transform_matrix_offset_center(matrix, x, y):
        o_x = tf.to_float(x) / 2 + 0.5
        o_y = tf.to_float(y) / 2 + 0.5
        offset_matrix = [[1, 0, o_x], [0, 1, o_y], [0, 0, 1]]
        reset_matrix = [[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]]
        transform_matrix = tf.matmul(
            tf.matmul(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    @staticmethod
    def _transform_matrices_to_flat(transform_matrices):
        # Flatten each matrix.
        transforms = array_ops.reshape(
            transform_matrices, constant_op.constant([-1, 9]))
        # Divide each matrix by the last entry (normally 1).
        transforms /= transforms[:, 8:9]
        return transforms[:, :8]

    def _augment_image(self, img):
        dims = tf.shape(img)
        transform_matrix = self._transform_matrix_offset_center(
            self.shear_matrix, dims[0], dims[1])
        transforms = self._transform_matrices_to_flat(transform_matrix)
        func = tf.contrib.image.transform(img, transforms)
        img = self.sometimes(self.p, func, img)
        return img


class StretchOrContract(Augmenter):
    """Stretch/contract the image."""

    def __init__(self, p, minval=0.9, maxval=1.1):
        self.p = p
        self.stretch_pct = tf.random_uniform([1], minval=minval, maxval=maxval)

    def _augment_image(self, img):
        dims = tf.shape(img)
        dims_float = tf.cast(tf.shape(img), dtype=tf.float32)
        zdims = tf.concat([
            tf.cast(tf.ceil(self.stretch_pct * dims_float[0]), dtype=tf.int32),
            tf.cast(tf.ceil(self.stretch_pct * dims_float[1]), dtype=tf.int32),
        ], 0)
        interp = tf.image.ResizeMethod.BICUBIC
        func = tf.image.resize_images(img, zdims, method=interp)
        func = tf.image.resize_image_with_crop_or_pad(func, dims[0], dims[1])
        img = self.sometimes(self.p, func, img)
        return img
