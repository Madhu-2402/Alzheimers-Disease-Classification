import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from albumentations import ElasticTransform, Compose, RandomBrightnessContrast, HueSaturationValue, GaussNoise
from PIL import Image

# Define source and target directories
dataset_dir = 'F:\\project\\Dataset\\OriginalDataset'
augmented_dataset = 'F:\\project\\Dataset\\Augmented_dataset'

# Create target directory if it doesn't exist
if not os.path.exists(augmented_dataset):
    os.makedirs(augmented_dataset)

# Define the augmentation pipeline for MRI
elastic_transform = Compose([
    ElasticTransform(alpha=1.0, sigma=50.0, always_apply=True),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5),
    HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, always_apply=False, p=0.5),
    GaussNoise(var_limit=(10.0, 50.0), always_apply=False, p=0.5)
])

def pad_image(img_array, pad_size=10):
    top = bottom = pad_size
    left = right = pad_size
    padded_image = np.pad(img_array, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=0)
    return padded_image

def resize_image(image, target_size):
    img = Image.fromarray(image)
    img = img.resize(target_size, Image.BILINEAR)  # or use other resampling methods
    return np.array(img)

# Create a list of subdirectories
with os.scandir(dataset_dir) as entries:
    sub_directories = [entry.name for entry in entries if entry.is_dir()]

for subdir in sub_directories:
    subdir_path = os.path.join(dataset_dir, subdir)
    target_subdir_path = os.path.join(augmented_dataset, subdir)

    # Create corresponding subdirectory in the target directory
    if not os.path.exists(target_subdir_path):
        os.makedirs(target_subdir_path)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.05,
        brightness_range=[0.9, 1.1],
        rescale=1./255,
        fill_mode='nearest'
    )

    for file_name in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file_name)

        if os.path.isfile(file_path):
            try:
                # Load image
                img = load_img(file_path)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = pad_image(img_array[0])  # Pad image before augmentation

                for i, batch in enumerate(datagen.flow(np.expand_dims(img_array, axis=0), batch_size=1)):
                    # Apply transformations
                    batch_img = batch[0]
                    batch_img = np.uint8(batch_img * 255)  # Convert from [0,1] to [0,255] for albumentations
                    augmented = elastic_transform(image=batch_img)
                    augmented_img = augmented['image']
                    augmented_img = resize_image(augmented_img, (img_array.shape[0], img_array.shape[1]))  # Resize image

                    # Save augmented image
                    augmented_img_path = os.path.join(target_subdir_path, f"{os.path.splitext(file_name)[0]}_aug_{i}.jpg")
                    augmented_img = array_to_img(augmented_img)
                    augmented_img.save(augmented_img_path)

                    if i >= 10:
                        break
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

print("Augmentation complete!")
