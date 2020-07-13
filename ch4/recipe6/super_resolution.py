import pathlib
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import *


def build_srcnn(height, width, depth):
    input = Input(shape=(height, width, depth))

    x = Conv2D(filters=64, kernel_size=(9, 9),
               kernel_initializer='he_normal')(input)
    x = ReLU()(x)
    x = Conv2D(filters=32, kernel_size=(1, 1),
               kernel_initializer='he_normal')(x)
    x = ReLU()(x)
    output = Conv2D(filters=depth, kernel_size=(5, 5),
                    kernel_initializer='he_normal')(x)

    return Model(input, output)


def resize_image(image_array, factor):
    original_image = Image.fromarray(image_array)
    new_size = np.array(original_image.size) * factor
    new_size = new_size.astype(np.int32)
    new_size = tuple(new_size)

    resized = original_image.resize(new_size)
    resized = img_to_array(resized)
    resized = resized.astype(np.uint8)

    return resized


def tight_crop_image(image):
    height, width = image.shape[:2]
    width -= int(width % SCALE)
    height -= int(height % SCALE)

    return image[:height, :width]


def downsize_upsize_image(image):
    scaled = resize_image(image, 1.0 / SCALE)
    scaled = resize_image(scaled, SCALE / 1.0)

    return scaled


def crop_input(image, x, y):
    y_slice = slice(y, y + INPUT_DIM)
    x_slice = slice(x, x + INPUT_DIM)

    return image[y_slice, x_slice]


def crop_output(image, x, y):
    y_slice = slice(y + PAD, y + PAD + LABEL_SIZE)
    x_slice = slice(x + PAD, x + PAD + LABEL_SIZE)

    return image[y_slice, x_slice]


SEED = 999
np.random.seed(SEED)
SUBSET_SIZE = 1500
file_patten = (pathlib.Path.home() / '.keras' / 'datasets' /
               'dogscats' / 'images' / '*.png')
file_pattern = str(file_patten)
dataset_paths = [*glob(file_pattern)]
dataset_paths = np.random.choice(dataset_paths, SUBSET_SIZE)

SCALE = 2.0
INPUT_DIM = 33
LABEL_SIZE = 21
PAD = int((INPUT_DIM - LABEL_SIZE) / 2.0)
STRIDE = 14

data = []
labels = []
for image_path in dataset_paths:
    image = load_img(image_path)
    image = img_to_array(image)
    image = image.astype(np.uint8)

    image = tight_crop_image(image)
    scaled = downsize_upsize_image(image)

    height, width = image.shape[:2]

    for y in range(0, height - INPUT_DIM + 1, STRIDE):
        for x in range(0, width - INPUT_DIM + 1, STRIDE):
            crop = crop_input(scaled, x, y)
            target = crop_output(image, x, y)

            data.append(crop)
            labels.append(target)

data = np.array(data)
labels = np.array(labels)

EPOCHS = 12

optimizer = Adam(lr=1e-3, decay=1e-3 / EPOCHS)
model = build_srcnn(INPUT_DIM, INPUT_DIM, 3)
model.compile(loss='mse', optimizer=optimizer)

BATCH_SIZE = 64
model.fit(data, labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Test on image
image = load_img('ch4/recipe6/dogs.jpg')
image = img_to_array(image)
image = image.astype(np.uint8)

image = tight_crop_image(image)
scaled = downsize_upsize_image(image)

plt.title('Low resolution image (Downsize + Upsize)')
plt.imshow(scaled)
plt.show()

output = np.zeros(scaled.shape)
height, width = output.shape[:2]

for y in range(0, height - INPUT_DIM + 1, LABEL_SIZE):
    for x in range(0, width - INPUT_DIM + 1, LABEL_SIZE):
        crop = crop_input(scaled, x, y)

        image_batch = np.expand_dims(crop, axis=0)
        prediction = model.predict(image_batch)
        new_shape = (LABEL_SIZE, LABEL_SIZE, 3)
        prediction = prediction.reshape(new_shape)

        output_y_slice = slice(y + PAD, y + PAD + LABEL_SIZE)
        output_x_slice = slice(x + PAD, x + PAD + LABEL_SIZE)
        output[output_y_slice, output_x_slice] = prediction

plt.title('Super resolution result (SRCNN output)')
plt.imshow(output / 255)
plt.show()
