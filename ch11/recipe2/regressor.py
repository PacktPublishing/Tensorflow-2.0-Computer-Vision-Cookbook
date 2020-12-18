import csv
import pathlib

import numpy as np
from autokeras import ImageRegressor
from tensorflow.keras.preprocessing.image import *


def load_mapping(csv_path, faces_path):
    mapping = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            file_name = line["file_name"].rsplit(".")[0]
            key = f'{faces_path}/{file_name}.jpg_face.jpg'
            mapping[key] = int(line['real_age'])

    return mapping


def get_images_and_labels(mapping):
    images = []
    labels = []
    for image_path, label in mapping.items():
        try:
            image = load_img(image_path, target_size=(64, 64))
            image = img_to_array(image)

            images.append(image)
            labels.append(label)
        except FileNotFoundError:
            continue

    return (np.array(images) - 127.5) / 127.5, \
           np.array(labels).astype('float32')

base_path = (pathlib.Path.home() / '.keras' / 'datasets' /
             'appa-real-release')
train_csv_path = str(base_path / 'gt_train.csv')
test_csv_path = str(base_path / 'gt_test.csv')
val_csv_path = str(base_path / 'gt_valid.csv')

train_faces_path = str(base_path / 'train')
test_faces_path = str(base_path / 'test')
val_faces_path = str(base_path / 'valid')

train_mapping = load_mapping(train_csv_path, train_faces_path)
test_mapping = load_mapping(test_csv_path, test_faces_path)
val_mapping = load_mapping(val_csv_path, val_faces_path)

X_train, y_train = get_images_and_labels(train_mapping)
X_test, y_test = get_images_and_labels(test_mapping)
X_val, y_val = get_images_and_labels(val_mapping)

EPOCHS = 15
regressor = ImageRegressor(seed=9,
                           max_trials=10,
                           optimizer='adam')
regressor.fit(X_train, y_train,
              epochs=EPOCHS,
              validation_data=(X_val, y_val),
              verbose=2)
print(regressor.evaluate(X_test, y_test))
