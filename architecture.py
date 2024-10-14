# Author: Jacob Dawson
#
# This file does two things:
# 1. We define our constants
# 2. We define our model architecture
# So, we can play around with architectural decisions before we start training

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# CONSTANTS!
seed = 8
num_channels = 3
batch_size = 32
image_size = 500
train_imgs_folder = "train_imgs/"

initializer = keras.initializers.RandomNormal(seed=seed)


def downsample(input, filters, size=2, stride=1, apply_batchnorm=True):
    # 3 convolutional layers
    conv1 = keras.layers.Conv2D(
        filters,
        kernel_size=size,
        strides=stride,
        padding="same",
        kernel_initializer=initializer,
        activation="selu",
    )(input)
    conv2 = keras.layers.Conv2D(
        filters,
        kernel_size=size,
        strides=stride,
        padding="same",
        kernel_initializer=initializer,
        activation="selu",
    )(conv1)
    conv3 = keras.layers.Conv2D(
        filters,
        kernel_size=size,
        strides=stride,
        padding="same",
        kernel_initializer=initializer,
        activation="selu",
    )(conv2)

    # concatenate those together
    out = keras.layers.Concatenate()([conv1, conv2, conv3])

    # then downsample. Couldn't decide on one method, so do both!
    maxpool_downsample = keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid"
    )(out)
    conv_downsample = keras.layers.Conv2D(
        filters,
        kernel_size=(2, 2),
        strides=2,
        padding="valid",
        kernel_initializer=initializer,
        activation="selu",
    )(out)
    out = keras.layers.Concatenate()([maxpool_downsample, conv_downsample])

    # batchnorm if we want
    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    return out


def model():
    input = keras.layers.Input(shape=(image_size, image_size, num_channels), dtype=tf.float32)
    out = downsample(input=input, filters=16, size=3, apply_batchnorm=False)
    out = downsample(input=out, filters=32, size=3)
    out = downsample(input=out, filters=64, size=3)
    out = downsample(input=out, filters=128, size=3)
    out = keras.layers.Conv2D(
        256,
        kernel_size=(4, 4),
        strides=4,
        padding="valid",
        kernel_initializer=initializer,
        activation="selu",
    )(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Conv2D(
        512,
        kernel_size=(4, 4),
        strides=4,
        padding="valid",
        kernel_initializer=initializer,
        activation="selu",
    )(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(units=200)(out)
    return keras.Model(inputs=input, outputs=out, name="bird_classifier")


if __name__ == "__main__":
    # display the model.
    m = model()
    m.summary()
    # this next line require graphviz, the bane of my existence
    # keras.utils.plot_model(m, to_file='bird_classifier_plot.png', show_shapes=True, show_layer_names=False, show_layer_activations=True, expand_nested=True)

    # you can use these for reverance if you want:
    # vgg16 = keras.applications.VGG16()
    # vgg16.summary()
    # incresnet = keras.applications.InceptionResNetV2()
    # incresnet.summary()
    # resnet = keras.applications.InceptionV3()
    # resnet.summary()
