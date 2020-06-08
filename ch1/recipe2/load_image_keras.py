import glob
import os
import tarfile

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.utils import get_file

DATASET_URL = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y'
DATA_NAME = 'cinic10'
FILE_EXTENSION = 'tar.gz'
FILE_NAME = '.'.join([DATA_NAME, FILE_EXTENSION])

# Downloading the data.
downloaded_file_location = get_file(origin=DATASET_URL, fname=FILE_NAME, extract=False)

data_directory, _ = downloaded_file_location.rsplit(os.path.sep, maxsplit=1)
data_directory = os.path.sep.join([data_directory, DATA_NAME])
tar = tarfile.open(downloaded_file_location)

if not os.path.exists(data_directory):
    tar.extractall(data_directory)

print(f'Data downloaded to {data_directory}')
data_pattern = os.path.sep.join([data_directory, '*/*/*.png'])

image_paths = list(glob.glob(data_pattern))
print(f'Sample image path: {image_paths[0]}')

# Load a single image
sample_image = load_img(image_paths[0])
print(f'Image type: {type(sample_image)}')
print(f'Image format: {sample_image.format}')
print(f'Image mode: {sample_image.mode}')
print(f'Image size: {sample_image.size}')

# Convert image to array
sample_image_array = img_to_array(sample_image)
print(f'Image array shape: {sample_image_array.shape}')
plt.imshow(sample_image_array / 255.0)

# Load a a batch of images.
scale_factor = 1.0 / 255.0
image_generator = ImageDataGenerator(horizontal_flip=True, rescale=scale_factor)

iterator = (image_generator
            .flow_from_directory(directory=data_directory,
                                 batch_size=10))
for batch, _ in iterator:
    plt.figure(figsize=(5, 5))
    for index, image in enumerate(batch, start=1):
        ax = plt.subplot(5, 5, index)
        plt.imshow(image)
        plt.axis('off')

    plt.show()
    break
