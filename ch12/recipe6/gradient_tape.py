import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist as fm
from tensorflow.keras.layers import *
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


def load_dataset():
    (X_train, y_train), (X_test, y_test) = fm.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape grayscale to include channel dimension.
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)


def build_network():
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(filters=20,
               kernel_size=(5, 5),
               padding='same',
               strides=(1, 1))(input_layer)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2),
                     strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=50,
               kernel_size=(5, 5),
               padding='same',
               strides=(1, 1))(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2),
                     strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(units=500)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)

    x = Dense(10)(x)
    output = Softmax()(x)

    return Model(inputs=input_layer, outputs=output)


def training_step(X, y, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = categorical_crossentropy(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,
                                  model.trainable_variables))


BATCH_SIZE = 256
EPOCHS = 100

(X_train, y_train), (X_test, y_test) = load_dataset()

optimizer = RMSprop()
model = build_network()

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    start = time.time()
    for i in range(int(len(X_train) / BATCH_SIZE)):
        X_batch = X_train[i * BATCH_SIZE:
                          i * BATCH_SIZE + BATCH_SIZE]
        y_batch = y_train[i * BATCH_SIZE:
                          i * BATCH_SIZE + BATCH_SIZE]

        training_step(X_batch, y_batch, model, optimizer)

    elapsed = time.time() - start
    print(f'\tElapsed time: {elapsed:.2f} seconds.')

model.compile(loss=categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])
results = model.evaluate(X_test, y_test)

print(f'Loss: {results[0]}, Accuracy: {results[1]}')
