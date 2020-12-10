import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import fashion_mnist as fm
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def load_dataset():
    (X_train, y_train), (X_test, y_test) = fm.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    (X_train, X_val,
     y_train, y_val) = train_test_split(X_train, y_train,
                                        train_size=0.8)

    train_ds = (tf.data.Dataset
                .from_tensor_slices((X_train, y_train)))
    val_ds = (tf.data.Dataset
              .from_tensor_slices((X_val, y_val)))
    test_ds = (tf.data.Dataset
               .from_tensor_slices((X_test, y_test)))

    train_ds = (train_ds
                .shuffle(buffer_size=BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .prefetch(buffer_size=BUFFER_SIZE))
    val_ds = (val_ds
              .batch(BATCH_SIZE)
              .prefetch(buffer_size=BUFFER_SIZE))
    test_ds = test_ds.batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds


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


def train_and_checkpoint(checkpointer):
    train_dataset, val_dataset, test_dataset = load_dataset()

    model = build_network()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=EPOCHS,
              callbacks=[checkpointer])


BATCH_SIZE = 256
BUFFER_SIZE = 1024
EPOCHS = 100

# Checkpoint saving all snapshots.
print('Running experiment 1: Saving all checkpoints.')
checkpoint_pattern = (
    'save_all/model-ep{epoch:03d}-loss{loss:.3f}'
    '-val_loss{val_loss:.3f}.h5')
checkpoint = ModelCheckpoint(checkpoint_pattern,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=False,
                             mode='min')
train_and_checkpoint(checkpoint)

# Checkpoint by saving the best only.
print('Running experiment 2: Saving best only.')
checkpoint_pattern = (
    'best_only/model-ep{epoch:03d}-loss{loss:.3f}'
    '-val_loss{val_loss:.3f}.h5')
checkpoint = ModelCheckpoint(checkpoint_pattern,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
train_and_checkpoint(checkpoint)

# Checkpoint by overwriting the best.
print('Running experiment 3: Overwriting best model.')
checkpoint_pattern = 'overwrite/model.h5'
checkpoint = ModelCheckpoint(checkpoint_pattern,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
train_and_checkpoint(checkpoint)