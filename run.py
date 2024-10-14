# Author: Jacob Dawson

from architecture import *

physical_devices = tf.config.experimental.list_physical_devices("GPU")
num_gpus = len(physical_devices)
print(f"Number of GPUs available: {num_gpus}")

directories = sorted(
    [
        d
        for d in os.listdir("train_imgs\CUB_200_2011\images")
        if os.path.isdir(os.path.join("train_imgs\CUB_200_2011\images", d))
    ]
)
# print(directories)


# took this function from here https://www.kaggle.com/code/ikkiocean/bird-species-classification-using-dl/notebook
# because I've never done image masking before.
def mask_image(directories):
    os.makedirs("Masked_Images", exist_ok=True)

    for directory in directories:
        img_directory = f"train_imgs\CUB_200_2011\images\{directory}"
        print(img_directory)
        img_files = sorted(os.listdir(img_directory))
        jpg_files = [img for img in img_files if img.endswith(".jpg")]
        seg_directory = f"train_imgs\segmentations\{directory}"
        seg_files = sorted(os.listdir(seg_directory))
        png_files = [img for img in seg_files if img.endswith(".png")]
        jpg_files = sorted(jpg_files)
        png_files = sorted(png_files)
        indexes = np.arange(len(jpg_files))
        np.random.shuffle(indexes)

        # Calculate the split point for 80:20
        split_point = int(0.8 * len(jpg_files))

        # Divide the indexes into 80:20
        train_indexes = indexes[:split_point]
        test_indexes = indexes[split_point:]

        train_split_point = int(0.75 * len(train_indexes))
        train_subset = train_indexes[:train_split_point]
        validation_subset = train_indexes[train_split_point:]

        print("Train indexes:", train_subset)
        print("Validation indexes:", validation_subset)
        print("Test indexes:", test_indexes)
        split_indexes = [train_subset, validation_subset, test_indexes]
        split_dir = ["train", "valid", "test"]
        jpg_array = np.array(jpg_files)
        png_array = np.array(png_files)
        for i in range(3):
            masked_image_count = 0
            for jpg_file, png_file in zip(
                jpg_array[split_indexes[i]], png_array[split_indexes[i]]
            ):
                # Load the original image and mask using Pillow
                image = Image.open(
                    f"train_imgs\CUB_200_2011\images\{directory}\{jpg_file}"
                )
                mask = Image.open(
                    f"train_imgs\segmentations\{directory}\{png_file}"
                ).convert(
                    "L"
                )  # Convert mask to grayscale

                # Ensure the mask has the same size as the image
                mask = mask.resize(image.size)

                # Convert the images to NumPy arrays
                image_array = np.array(image)
                mask_array = np.array(mask)

                # Normalize the mask to be in the range of [0, 1]
                mask_array = mask_array / 255.0

                # Ensure the mask has the correct shape (broadcastable)
                if len(image_array.shape) == 3:
                    mask_array = np.expand_dims(mask_array, axis=-1)

                # Apply the mask to the image
                masked_image_array = image_array * mask_array

                # Convert the result back to a PIL Image
                masked_image = Image.fromarray(np.uint8(masked_image_array))

                # Optionally save the result
                os.makedirs(f"Masked_Images/{split_dir[i]}/{directory}", exist_ok=True)
                masked_image.save(
                    f"Masked_Images/{split_dir[i]}/{directory}/{jpg_file}"
                )
                masked_image_count += 1
                print(
                    f"Masking {jpg_file} {split_dir[i]} completed - {masked_image_count}"
                )


# mask_image(directories)

# give each directory (bird species) a number:
number_of_bird = 0
bird_names_to_numbers = {}
bird_numbers_to_names = {}
bird_name_number_tuples = []
for dir in os.listdir("Masked_Images/train/"):
    name_of_bird = dir.split(".")[1]
    bird_names_to_numbers[name_of_bird] = number_of_bird
    bird_numbers_to_names[number_of_bird] = name_of_bird
    bird_name_number_tuples.append((name_of_bird, number_of_bird))


# make functions for reading from our train, val, and test directories.
# Repeated because tf requires this to take no arguments, for some reason.
def train_gen(type_dir="Masked_Images/train"):
    for dir in os.listdir(f"{type_dir}/"):
        name_of_bird = dir.split(".")[1]
        number_of_bird = bird_names_to_numbers[name_of_bird]
        for img in os.listdir(f"{type_dir}/{dir}/"):
            loaded_img = np.array(Image.open(f"{type_dir}/{dir}/{img}"))
            # print(type(loaded_img),loaded_img.shape, loaded_img.dtype, type(number_of_bird))
            # <class 'numpy.ndarray'> (223, 320, 3) uint8 <class 'int'>
            yield loaded_img, number_of_bird


def test_gen(type_dir="Masked_Images/test"):
    for dir in os.listdir(f"{type_dir}/"):
        name_of_bird = dir.split(".")[1]
        number_of_bird = bird_names_to_numbers[name_of_bird]
        for img in os.listdir(f"{type_dir}/{dir}/"):
            loaded_img = np.array(Image.open(f"{type_dir}/{dir}/{img}"))
            # print(type(loaded_img),loaded_img.shape, loaded_img.dtype, type(number_of_bird))
            # <class 'numpy.ndarray'> (223, 320, 3) uint8 <class 'int'>
            yield loaded_img, number_of_bird


def val_gen(type_dir="Masked_Images/val"):
    for dir in os.listdir(f"{type_dir}/"):
        name_of_bird = dir.split(".")[1]
        number_of_bird = bird_names_to_numbers[name_of_bird]
        for img in os.listdir(f"{type_dir}/{dir}/"):
            loaded_img = np.array(
                Image.open(f"{type_dir}/{dir}/{img}"), dtype=np.float32
            )
            yield loaded_img, number_of_bird


def rescale(image, label):
    # TODO: fix this!
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    image = image / 255.0
    return image, label


def augment(image, label):
    # TODO: fix this!
    print(tf.shape(image), tf.shape(label))
    image, label = rescale(image, label)
    image = tf.image.stateless_random_crop(
        image, size=[image_size, image_size, num_channels], seed=seed
    )
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=seed)
    image = tf.clip_by_value(image, 0, 1)
    return image, label


# turn those generators into datasets, which we can then batch and augment and whatnot
train_ds = (
    tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(
                shape=(image_size, image_size, num_channels), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    .map(rescale, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_generator(
        test_gen,
        output_signature=(
            tf.TensorSpec(
                shape=(image_size, image_size, num_channels), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    .map(rescale, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec(
                shape=(image_size, image_size, num_channels), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    .map(rescale, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# print(f"Cardinality of train set: {train_ds.cardinality()}")
# print(f"Cardinality of test set: {test_ds.cardinality()}")
# print(f"Cardinality of val set: {val_ds.cardinality()}")

bird_model = model()
bird_model.summary()
"""
bird_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = bird_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)]
)
"""
