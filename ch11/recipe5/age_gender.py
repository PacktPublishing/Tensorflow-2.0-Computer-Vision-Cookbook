import csv
import os
import pathlib
from glob import glob

import cv2
import imutils
import numpy as np
from autokeras import *
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import *

base_path = (pathlib.Path.home() / '.keras' / 'datasets' /
             'adience')
folds_path = str(base_path / 'folds')

AGE_BINS = [(0, 2), (4, 6), (8, 13), (15, 20), (25, 32),
            (38, 43), (48, 53), (60, 99)]


def age_to_bin(age):
    age = age.replace('(', '').replace(')', '').split(',')
    lower, upper = [int(x.strip()) for x in age]

    for bin_low, bin_up in AGE_BINS:
        if lower >= bin_low and upper <= bin_up:
            label = f'{bin_low}_{bin_up}'
            return label


def rectangle_area(r):
    return (r[2] - r[0]) * (r[3] - r[1])


def plot_face(image, age_gender, detection):
    frame_x, frame_y, frame_width, frame_height = detection
    cv2.rectangle(image,
                  (frame_x, frame_y),
                  (frame_x + frame_width,
                   frame_y + frame_height),
                  color=(0, 255, 0),
                  thickness=2)
    cv2.putText(image,
                age_gender,
                (frame_x, frame_y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.45,
                color=(0, 255, 0),
                thickness=2)

    return image


def predict(model, roi):
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype('float32') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi)[0]
    return predictions


images = []
ages = []
genders = []
folds_pattern = os.path.sep.join([folds_path, '*.txt'])
for fold_path in glob(folds_pattern):
    with open(fold_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for line in reader:
            if ((line['age'][0] != '(') or
                    (line['gender'] not in {'m', 'f'})):
                continue

            age_label = age_to_bin(line['age'])
            if age_label is None:
                continue

            aligned_face_file = (f'landmark_aligned_face.'
                                 f'{line["face_id"]}.'
                                 f'{line["original_image"]}')
            image_path = os.path.sep.join([str(base_path),
                                           line["user_id"],
                                           aligned_face_file])

            image = load_img(image_path, target_size=(64, 64))
            image = img_to_array(image)

            images.append(image)
            ages.append(age_label)
            genders.append(line['gender'])

age_images = np.array(images).astype('float32') / 255.0
gender_images = np.copy(images)

gender_enc = LabelEncoder()
age_enc = LabelEncoder()
gender_labels = gender_enc.fit_transform(genders)
age_labels = age_enc.fit_transform(ages)

EPOCHS = 100
MAX_TRIALS = 10

if os.path.exists('age_model.h5'):
    age_model = load_model('age_model.h5')
else:
    age_clf = ImageClassifier(seed=9,
                              max_trials=MAX_TRIALS,
                              project_name='age_clf',
                              overwrite=True)
    age_clf.fit(age_images, age_labels, epochs=EPOCHS)
    age_model = age_clf.export_model()
    age_model.save('age_model.h5')

if os.path.exists('gender_model.h5'):
    gender_model = load_model('gender_model.h5')
else:
    gender_clf = ImageClassifier(seed=9,
                                 max_trials=MAX_TRIALS,
                                 project_name='gender_clf',
                                 overwrite=True)
    gender_clf.fit(gender_images, gender_labels,
                   epochs=EPOCHS)
    gender_model = gender_clf.export_model()
    gender_model.save('gender_model.h5')

image = cv2.imread('woman.jpg')

cascade_file = 'resources/haarcascade_frontalface_default.xml'
det = cv2.CascadeClassifier(cascade_file)

image = imutils.resize(image, width=380)
copy = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detections = \
    det.detectMultiScale(gray,
                         scaleFactor=1.1,
                         minNeighbors=5,
                         minSize=(35, 35),
                         flags=cv2.CASCADE_SCALE_IMAGE)

if len(detections) > 0:
    detections = sorted(detections, key=rectangle_area)
    best_detection = detections[-1]

    (frame_x, frame_y,
     frame_width, frame_height) = best_detection

    roi = image[frame_y:frame_y + frame_height,
          frame_x:frame_x + frame_width]

    age_pred = predict(age_model, roi).argmax()
    age = age_enc.inverse_transform([age_pred])[0]

    gender_pred = predict(gender_model, roi).argmax()
    gender = gender_enc.inverse_transform([gender_pred])[0]

    clone = plot_face(copy,
                      f'Gender: {gender} - Age: {age}',
                      best_detection)

    cv2.imshow('Result', copy)
    cv2.waitKey(0)
