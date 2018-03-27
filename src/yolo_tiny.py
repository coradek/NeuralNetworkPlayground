"""
Implementation of tiny YOLO in Keras
YOLO- You Only Look Once: Unified, Real-Time Object Detection
https://arxiv.org/abs/1506.02640

http://machinethink.net/blog/object-detection-with-yolo/
https://github.com/joycex99/tiny-yolo-keras
https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb

YOLO uses convolution to reduce an image to a grid of
sub-sections (by default a 13x13 squares). For each subsection,
the model returns series of predictions (one for each of 5 anchor box shapes).
each prediction includes
    - the probability of a target object's center being in that subsection
    - the likelyhood of the object being in each of the possible classes
    - the dimensions of a bounding box for that object
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import SGD, Adam


##--------------------
## TODO: add args for
##       image input size
##       anchorbox num/shapes
##       grid dimensions
##--------------------


class YoloTiny(object):
    """
    Tiny Version of the You Only Look Once
    Unified, Real-Time Object Detection Architecture
    """
    def __init__(self,
                 img_dim=(416,416),
                 grid_dim=(13,13),
                 num_anchor_boxes=5,
                 num_classes=20):

        self.img_dim = img_dim     # Size of input images
        self.grid_dim = grid_dim   # Image subsections
        self.num_anchor_boxes = num_anchor_boxes
        self.num_classes = num_classes

        ## This is a training parameter - should be outside yolo class
        BATCH_SIZE = 8

        NORM_H, NORM_W = self.img_dim
        GRID_H, GRID_W = self.grid_dim
        BOX = self.num_anchor_boxes
        ORIG_CLASS = self.num_classes

        self.model = self._build_model()


    def _build_model():
        model = Sequential([
                ## Layer One (all layers follow this general structure)
                    Conv2D(16, (3,3),
                           strides=(1,1), padding='same',
                           use_bias=False, input_shape=(416,416,3))
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    MaxPooling2D(pool_size=(2, 2)),
                ## Layers 2-6
                    conv_batchnorm_activate_pool(32),
                    conv_batchnorm_activate_pool(64),
                    conv_batchnorm_activate_pool(128),
                    conv_batchnorm_activate_pool(256),
                    conv_batchnorm_activate_pool(512,
                                                 pool_stride=(1,1),
                                                 pool_padding="same"),
                ## Layers 7-8 do not have max pooling
                    conv_batchnorm_activate_pool(1024, pool=False),
                    conv_batchnorm_activate_pool(1024, pool=False),
                ## Layer 9
                    Conv2D(BOX * (4 + 1 + ORIG_CLASS),
                           (1, 1), strides=(1, 1),
                           kernel_initializer='he_normal'),
                    Activation('linear'),
                    Reshape((GRID_H, GRID_W, BOX, 4 + 1 + ORIG_CLASS)),
                ])
        return model



def custom_loss(y_true, y_pred):
    """
    The YOLO architecture is a straight forward CNN
    What makes it awesome is the special loss function which optimizes
    - the probability an object exists in a particular grid square
    - the likely hood of each class of object being detected
    - which boxes are most likely to contain a target object
    - the amount of overlap for all boxes containing the same
      class of object
    - and the amount of overlap between the boxes the model considers
      most likely to contain a target object and the actual boxes for an image

    all in one fell swoop
    """
    ### Adjust prediction
    # adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[:,:,:,:,:2])

    # adjust w and h
    pred_box_wh = tf.exp(y_pred[:,:,:,:,2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1,1,1,1,2]))

    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)

    # adjust probability
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])

    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
    print("Y_pred shape: {}".format(y_pred.shape))

    ### Adjust ground truth
    # adjust x and y
    center_xy = .5*(y_true[:,:,:,:,0:2] + y_true[:,:,:,:,2:4])
    center_xy = center_xy / np.reshape([(float(NORM_W)/GRID_W), (float(NORM_H)/GRID_H)], [1,1,1,1,2])
    true_box_xy = center_xy - tf.floor(center_xy)

    # adjust w and h
    true_box_wh = (y_true[:,:,:,:,2:4] - y_true[:,:,:,:,0:2])
    true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(NORM_W), float(NORM_H)], [1,1,1,1,2]))

    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    pred_box_area = pred_tem_wh[:,:,:,:,0] * pred_tem_wh[:,:,:,:,1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

    true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    true_box_area = true_tem_wh[:,:,:,:,0] * true_tem_wh[:,:,:,:,1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh

    intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]

    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
    best_box = tf.to_float(best_box)
    true_box_conf = tf.expand_dims(best_box * y_true[:,:,:,:,4], -1)

    # adjust confidence
    true_box_prob = y_true[:,:,:,:,5:]

    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
    print("Y_true shape: {}".format(y_true.shape))
    #y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)

    ### Compute the weights
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = SCALE_COOR * weight_coor

    weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf

    weight_prob = tf.concat(CLASS * [true_box_conf], 4)
    weight_prob = SCALE_PROB * weight_prob

    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)
    print("Weight shape: {}".format(weight.shape))

    ### Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, GRID_W*GRID_H*BOX*(4 + 1 + CLASS)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)

    return loss


def conv_batchnorm_activate_pool(conv_filters,
                                 pool=True,
                                 pool_stride=None,
                                 pool_padding='valid'):
    """return convolution, nomalization, activation and pooling layers

    All YOLO layers follow this structure with a few minor changes

    Conv2D
    BatchNormalization
    LeakyReLU
    MaxPooling2D (as needed)
    """
    layers = Sequential([
                Conv2D(conv_filters,
                       (3,3), strides=(1,1),
                       padding='same', use_bias=False)
                BatchNormalization(),
                LeakyReLU(alpha=0.1)
                ])
    if pool:
        layers.add(MaxPooling2D(pool_size=(2, 2),
                                strides=pool_stride,
                                padding=pool_padding))
    return layers
