import csv
import glob
import pathlib

import cv2
import imutils
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.utils import to_categorical

EMOTIONS = ['angry', 'scared', 'happy', 'sad', 'surprised',
            'neutral']

COLORS = {
    'angry': (0, 0, 255),
    'scared': (0, 128, 255),
    'happy': (0, 255, 255),
    'sad': (255, 0, 0),
    'surprised': (178, 255, 102),
    'neutral': (160, 160, 160)
}


def build_network(input_shape, classes):
    input = Input(shape=input_shape)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               kernel_initializer='he_normal')(input)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               kernel_initializer='he_normal',
               padding='same')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Flatten()(x)
    x = Dense(units=64,
              kernel_initializer='he_normal')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=64,
              kernel_initializer='he_normal')(x)
    x = ELU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=classes,
              kernel_initializer='he_normal')(x)
    output = Softmax()(x)

    return Model(input, output)


def load_dataset(dataset_path, classes):
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []

    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            label = int(line['emotion'])

            if label <= 1:
                label = 0  # This merges classes 1 and 0.

            if label > 0:
                label -= 1  # All classes start from 0.

            image = np.array(line['pixels'].split(' '),
                             dtype='uint8')
            image = image.reshape((48, 48))
            image = img_to_array(image)

            if line['Usage'] == 'Training':
                train_images.append(image)
                train_labels.append(label)
            elif line['Usage'] == 'PrivateTest':
                val_images.append(image)
                val_labels.append(label)
            else:
                test_images.append(image)
                test_labels.append(label)

    train_images = np.array(train_images)
    val_images = np.array(val_images)
    test_images = np.array(test_images)

    train_labels = to_categorical(np.array(train_labels),
                                  classes)
    val_labels = to_categorical(np.array(val_labels), classes)
    test_labels = to_categorical(np.array(test_labels),
                                 classes)

    return (train_images, train_labels), \
           (val_images, val_labels), \
           (test_images, test_labels)


def rectangle_area(r):
    return (r[2] - r[0]) * (r[3] - r[1])


def plot_emotion(emotions_plot, emotion, probability, index):
    w = int(probability * emotions_plot.shape[1])
    cv2.rectangle(emotions_plot,
                  (5, (index * 35) + 5),
                  (w, (index * 35) + 35),
                  color=COLORS[emotion],
                  thickness=-1)

    white = (255, 255, 255)
    text = f'{emotion}: {probability * 100:.2f}%'
    cv2.putText(emotions_plot,
                text,
                (10, (index * 35) + 23),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.45,
                color=white,
                thickness=2)

    return emotions_plot


def plot_face(image, emotion, detection):
    frame_x, frame_y, frame_width, frame_height = detection
    cv2.rectangle(image,
                  (frame_x, frame_y),
                  (frame_x + frame_width,
                   frame_y + frame_height),
                  color=COLORS[emotion],
                  thickness=2)
    cv2.putText(image,
                emotion,
                (frame_x, frame_y - 10),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.45,
                color=COLORS[emotion],
                thickness=2)

    return image


def predict_emotion(model, roi):
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi)[0]
    return predictions


checkpoints = sorted(list(glob.glob('./*.h5')), reverse=True)
if len(checkpoints) > 0:
    model = load_model(checkpoints[0])
else:
    base_path = (pathlib.Path.home() / '.keras' / 'datasets' /
                 'emotion_recognition' / 'fer2013')
    input_path = str(base_path / 'fer2013.csv')
    classes = len(EMOTIONS)

    (train_images, train_labels), \
    (val_images, val_labels), \
    (test_images, test_labels) = load_dataset(input_path,
                                              classes)

    model = build_network((48, 48, 1), classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.003),
                  metrics=['accuracy'])

    checkpoint_pattern = ('model-ep{epoch:03d}-loss{loss:.3f}'
                          '-val_loss{val_loss:.3f}.h5')
    checkpoint = ModelCheckpoint(checkpoint_pattern,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    BATCH_SIZE = 128
    train_augmenter = ImageDataGenerator(rotation_range=10,
                                         zoom_range=0.1,
                                         horizontal_flip=True,
                                         rescale=1. / 255.,
                                         fill_mode='nearest')
    train_gen = train_augmenter.flow(train_images,
                                     train_labels,
                                     batch_size=BATCH_SIZE)
    train_steps = len(train_images) // BATCH_SIZE

    val_augmenter = ImageDataGenerator(rescale=1. / 255.)
    val_gen = val_augmenter.flow(val_images,
                                 val_labels,
                                 batch_size=BATCH_SIZE)

    EPOCHS = 300
    model.fit(train_gen,
              steps_per_epoch=train_steps,
              validation_data=val_gen,
              epochs=EPOCHS,
              verbose=1,
              callbacks=[checkpoint])

    test_augmenter = ImageDataGenerator(rescale=1. / 255.)
    test_gen = test_augmenter.flow(test_images,
                                   test_labels,
                                   batch_size=BATCH_SIZE)
    test_steps = len(test_images) // BATCH_SIZE
    _, accuracy = model.evaluate(test_gen, steps=test_steps)

    print(f'Accuracy: {accuracy * 100}%')

video_path = 'emotions.mp4'
camera = cv2.VideoCapture(video_path)  # Pass 0 to use webcam

cascade_file = 'resources/haarcascade_frontalface_default.xml'
det = cv2.CascadeClassifier(cascade_file)
while True:
    frame_exists, frame = camera.read()

    if not frame_exists:
        break

    frame = imutils.resize(frame, width=380)
    emotions_plot = np.zeros_like(frame, dtype='uint8')
    copy = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = \
        det.detectMultiScale(gray,
                             scaleFactor=1.1,
                             minNeighbors=5,
                             minSize=(35, 35),
                             flags=cv2.CASCADE_SCALE_IMAGE)

    if len(detections) > 0:
        detections = sorted(detections,
                            key=rectangle_area)
        best_detection = detections[-1]

        (frame_x, frame_y,
         frame_width, frame_height) = best_detection

        roi = gray[frame_y:frame_y + frame_height,
                   frame_x:frame_x + frame_width]
        predictions = predict_emotion(model, roi)
        label = EMOTIONS[predictions.argmax()]

        for i, (emotion, probability) in \
                enumerate(zip(EMOTIONS, predictions)):
            emotions_plot = plot_emotion(emotions_plot,
                                         emotion,
                                         probability,
                                         i)

        clone = plot_face(copy, label, best_detection)

    cv2.imshow('Face & emotions',
               np.hstack([copy, emotions_plot]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
