# Author: Jacob Dawson
#
# This file does two things:
# 1. We define our constants
# 2. We define our model architecture
# So, we can play around with architectural decisions before we start training

from tensorflow import keras
import tensorflow as tf

# CONSTANTS!
seed = 8
num_channels = 3
batch_size = 32
image_size = 0
train_imgs_folder = "train_imgs/"

initializer = keras.initializers.RandomNormal(seed=seed)


def downsample(input, filters, size=2, stride=2, apply_batchnorm=True):
    out = keras.layers.Conv2D(
        filters,
        kernel_size=size,
        strides=stride,
        padding="same",
        kernel_initializer=initializer,
    )(input)
    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation("selu")(out)
    return out


def model():
    input = keras.layers.Input(shape=(None, None, num_channels), dtype=tf.float32)
    scale = keras.layers.Rescaling(1.0 / 255.0, offset=0)(input)
    d1 = downsample(scale, 8, 2, apply_batchnorm=False)
    d2 = downsample(input=d1, filters=16, size=2)
    d3 = downsample(input=d2, filters=32, size=2)
    d4 = downsample(input=d3, filters=64, size=2)
    out = keras.layers.Rescaling(255.0)(d4)
    return keras.Model(inputs=input, outputs=out, name="bird_classifier")


if __name__ == "__main__":
    # display the model.
    m = model()
    m.summary()
    # this next line require graphviz, the bane of my existence
    # keras.utils.plot_model(m, to_file='bird_classifier_plot.png', show_shapes=True, show_layer_names=False, show_layer_activations=True, expand_nested=True)
