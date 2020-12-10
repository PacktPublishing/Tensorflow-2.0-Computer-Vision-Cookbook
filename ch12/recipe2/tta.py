import os
import pathlib
from glob import glob

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import *


def load_images_and_labels(image_paths, target_size=(64, 64)):
    images = []
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)

        label = image_path.split(os.path.sep)[-2]

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


def build_network(width, height, depth, classes):
    input_layer = Input(shape=(width, height, depth))

    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same')(input_layer)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Flatten()(x)
    x = Dense(units=512)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.25)(x)

    x = Dense(units=classes)(x)
    output = Softmax()(x)

    return Model(input_layer, output)


def flip_augment(image, num_test=10):
    augmented = []
    for i in range(num_test):
        should_flip = np.random.randint(0, 2)
        if should_flip:
            flipped = np.fliplr(image.copy())
            augmented.append(flipped)
        else:
            augmented.append(image.copy())

    return np.array(augmented)


SEED = 84
np.random.seed(SEED)

base_path = (pathlib.Path.home() / '.keras' / 'datasets' /
             '101_ObjectCategories')
images_pattern = str(base_path / '*' / '*.jpg')
image_paths = [*glob(images_pattern)]
image_paths = [p for p in image_paths if
               p.split(os.path.sep)[-2] != 'BACKGROUND_Google']
CLASSES = {p.split(os.path.sep)[-2] for p in image_paths}

X, y = load_images_and_labels(image_paths)
X = X.astype('float') / 255.0
y = LabelBinarizer().fit_transform(y)

(X_train, X_test,
 y_train, y_test) = train_test_split(X, y,
                                     test_size=0.2,
                                     random_state=SEED)
BATCH_SIZE = 64
EPOCHS = 40

augmenter = ImageDataGenerator(horizontal_flip=True)

model = build_network(64, 64, 3, len(CLASSES))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_generator = augmenter.flow(X_train, y_train,
                                 BATCH_SIZE)
model.fit(train_generator,
          steps_per_epoch=len(X_train) // BATCH_SIZE,
          validation_data=(X_test, y_test),
          epochs=EPOCHS,
          verbose=2)

predictions = model.predict(X_test,
                            batch_size=BATCH_SIZE)
accuracy = accuracy_score(y_test.argmax(axis=1),
                          predictions.argmax(axis=1))
print(f'Accuracy, without TTA: {accuracy}')

predictions = []
NUM_TEST = 10
for index in range(len(X_test)):
    batch = flip_augment(X_test[index], NUM_TEST)
    sample_predictions = model.predict(batch)

    sample_predictions = np.argmax(
        np.sum(sample_predictions, axis=0))
    predictions.append(sample_predictions)

accuracy = accuracy_score(y_test.argmax(axis=1), predictions)
print(f'Accuracy with TTA: {accuracy}')
