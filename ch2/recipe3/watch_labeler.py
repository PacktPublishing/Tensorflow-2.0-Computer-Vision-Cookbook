import glob
import os
import pathlib
from csv import DictReader

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


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
    output = Activation('sigmoid')(x)

    return Model(input_layer, output)


def load_images_and_labels(image_paths, styles, target_size):
    images = []
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image_id = image_path.split(os.path.sep)[-1][:-4]

        image_style = styles[image_id]
        label = (image_style['gender'], image_style['usage'])

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


SEED = 999
np.random.seed(SEED)

base_path = (pathlib.Path.home() / '.keras' / 'datasets' /
             'fashion-product-images-small')
styles_path = str(base_path / 'styles.csv')
images_path_pattern = str(base_path / 'images/*.jpg')
image_paths = glob.glob(images_path_pattern)

with open(styles_path, 'r') as f:
    dict_reader = DictReader(f)
    STYLES = [*dict_reader]

    article_type = 'Watches'
    genders = {'Men', 'Women'}
    usages = {'Casual', 'Smart Casual', 'Formal'}
    STYLES = {style['id']: style
              for style in STYLES
              if (style['articleType'] == article_type and
                  style['gender'] in genders and
                  style['usage'] in usages)}

image_paths = [*filter(lambda p: p.split(os.path.sep)[-1][:-4]
                                 in STYLES.keys(),
                       image_paths)]

X, y = load_images_and_labels(image_paths, STYLES, (64, 64))
X = X.astype('float') / 255.0

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

(X_train, X_test,
 y_train, y_test) = train_test_split(X, y,
                                     stratify=y,
                                     test_size=0.2,
                                     random_state=SEED)
(X_train, X_valid,
 y_train, y_valid) = train_test_split(X_train, y_train,
                                      stratify=y_train,
                                      test_size=0.2,
                                      random_state=SEED)

model = build_network(width=64,
                      height=64,
                      depth=3,
                      classes=len(mlb.classes_))
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

BATCH_SIZE = 64
EPOCHS = 20
model.fit(X_train, y_train,
          validation_data=(X_valid, y_valid),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)

result = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print(f'Test accuracy: {result[1]}')

test_image = np.expand_dims(X_test[0], axis=0)
probabilities = model.predict(test_image)[0]

for label, p in zip(mlb.classes_, probabilities):
    print(f'{label}: {p * 100:.2f}%')

ground_truth_labels = np.expand_dims(y_test[0], axis=0)
ground_truth_labels = mlb.inverse_transform(ground_truth_labels)
print(f'Ground truth labels: {ground_truth_labels}')
