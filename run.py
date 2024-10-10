# Author: Jacob Dawson

from architecture import *

physical_devices = tf.config.experimental.list_physical_devices("GPU")
num_gpus = len(physical_devices)
print(f"Number of GPUs available: {num_gpus}")

# TODO:
# 1. fix the 'gpus not found' issue
# 2. figure out how to make a dataset that will traverse our train_imgs folder

"""
train_imgs = keras.utils.image_dataset_from_directory(
    "train_imgs/",
    labels=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(image_size, image_size),
    shuffle=True,
    interpolation="bilinear",
    seed=seed,
)
"""
