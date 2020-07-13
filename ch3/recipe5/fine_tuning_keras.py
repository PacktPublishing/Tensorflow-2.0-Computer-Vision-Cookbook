import os
import pathlib
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import *

SEED = 999


def build_network(base_model, classes):
    x = Flatten()(base_model.output)
    x = Dense(units=256)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=classes)(x)
    output = Softmax()(x)

    return output


def load_images_and_labels(image_paths,
                           target_size=(256, 256)):
    images = []
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)

        label = image_path.split(os.path.sep)[-2]

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


dataset_path = (pathlib.Path.home() / '.keras' / 'datasets' /
                'flowers17')
files_pattern = (dataset_path / 'images' / '*' / '*.jpg')
image_paths = [*glob(str(files_pattern))]
CLASSES = {p.split(os.path.sep)[-2] for p in image_paths}

X, y = load_images_and_labels(image_paths)
X = X.astype('float') / 255.0
y = LabelBinarizer().fit_transform(y)

(X_train, X_test,
 y_train, y_test) = train_test_split(X, y,
                                     test_size=0.2,
                                     random_state=SEED)

base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=Input(shape=(256, 256, 3)))

for layer in base_model.layers:
    layer.trainable = False

model = build_network(base_model, len(CLASSES))
model = Model(base_model.input, model)

BATCH_SIZE = 64
augmenter = ImageDataGenerator(rotation_range=30,
                               horizontal_flip=True,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.2,
                               zoom_range=0.2,
                               fill_mode='nearest')
train_generator = augmenter.flow(X_train, y_train, BATCH_SIZE)

WARMING_EPOCHS = 20
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-3),
              metrics=['accuracy'])
history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    epochs=WARMING_EPOCHS)
result = model.evaluate(X_test, y_test)
print(f'Test accuracy: {result[1]}')

for layer in base_model.layers[15:]:
    layer.trainable = True

EPOCHS = 50
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-3),
              metrics=['accuracy'])
history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS)
result = model.evaluate(X_test, y_test)
print(f'Test accuracy: {result[1]}')
