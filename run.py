# Author: Jacob Dawson

from architecture import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(physical_devices)
print(f"Number of GPUs available: {num_gpus}")
