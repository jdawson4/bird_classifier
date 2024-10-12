# Author: Jacob Dawson

from architecture import *

physical_devices = tf.config.experimental.list_physical_devices("GPU")
num_gpus = len(physical_devices)
print(f"Number of GPUs available: {num_gpus}")

# TODO:
# 1. fix the 'gpus not found' issue
# 2. figure out how to make a dataset that will traverse our train_imgs folder

directories=sorted([d for d in os.listdir("train_imgs\CUB_200_2011\images") if os.path.isdir(os.path.join("train_imgs\CUB_200_2011\images", d))])
#print(directories)

def mask_image(directories):
    os.makedirs('Masked_Images', exist_ok=True)
    
    for directory in directories:
        img_directory = f'train_imgs\CUB_200_2011\images\{directory}'
        print(img_directory)
        img_files = sorted(os.listdir(img_directory))
        jpg_files = [img for img in img_files if img.endswith('.jpg')]
        seg_directory = f'train_imgs\segmentations\{directory}' 
        seg_files = sorted(os.listdir(seg_directory))
        png_files = [img for img in seg_files if img.endswith('.png')]
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
        split_indexes = [train_subset,validation_subset,test_indexes]
        split_dir = ['train','valid','test']
        jpg_array = np.array(jpg_files)
        png_array = np.array(png_files)
        for i in range(3):
            masked_image_count = 0
            for jpg_file,png_file in zip(jpg_array[split_indexes[i]],png_array[split_indexes[i]]):
            # Load the original image and mask using Pillow
                image = Image.open(f'train_imgs\CUB_200_2011\images\{directory}\{jpg_file}')
                mask = Image.open(f'train_imgs\segmentations\{directory}\{png_file}').convert('L')  # Convert mask to grayscale

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
                os.makedirs(f'Masked_Images/{split_dir[i]}/{directory}', exist_ok=True)
                masked_image.save(f'Masked_Images/{split_dir[i]}/{directory}/{jpg_file}')
                masked_image_count += 1
                print(f'Masking {jpg_file} {split_dir[i]} completed - {masked_image_count}')

#mask_image(directories)

def train_gen():
    for dir in os.listdir("Masked_Images/train/"):
        print(dir)
        for img in os.listdir(f"Masked_Images/train/{dir}/"):
            print(img)
            return

train_gen()
#train_ds = tf.data.Dataset.

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
