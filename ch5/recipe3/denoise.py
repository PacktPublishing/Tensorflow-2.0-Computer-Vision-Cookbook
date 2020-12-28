import cv2
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import *


def build_autoencoder(input_shape=(28, 28, 1),
                      encoding_size=128,
                      alpha=0.2):
    inputs = Input(shape=input_shape)
    encoder = Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=2,
                     padding='same')(inputs)
    encoder = LeakyReLU(alpha=alpha)(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Conv2D(filters=64,
                     kernel_size=(3, 3),
                     strides=2,
                     padding='same')(encoder)
    encoder = LeakyReLU(alpha=alpha)(encoder)
    encoder = BatchNormalization()(encoder)

    encoder_output_shape = encoder.shape
    encoder = Flatten()(encoder)
    encoder_output = Dense(units=encoding_size)(encoder)

    encoder_model = Model(inputs, encoder_output)

    # Build decoder
    decoder_input = Input(shape=(encoding_size,))
    decoder_dense_units = np.prod(encoder_output_shape[1:])
    decoder = Dense(units=decoder_dense_units)(decoder_input)
    decoder = Reshape(tuple(encoder_output_shape[1:]))(decoder)

    decoder = Conv2DTranspose(filters=64,
                              kernel_size=(3, 3),
                              strides=2,
                              padding='same')(decoder)
    decoder = LeakyReLU(alpha=alpha)(decoder)
    decoder = BatchNormalization()(decoder)

    decoder = Conv2DTranspose(filters=32,
                              kernel_size=(3, 3),
                              strides=2,
                              padding='same')(decoder)
    decoder = LeakyReLU(alpha=alpha)(decoder)
    decoder = BatchNormalization()(decoder)

    decoder = Conv2DTranspose(filters=1,
                              kernel_size=(3, 3),
                              padding='same')(decoder)
    outputs = Activation('sigmoid')(decoder)

    decoder_model = Model(decoder_input, outputs)

    encoder_model_output = encoder_model(inputs)
    decoder_model_output = decoder_model(encoder_model_output)
    autoencoder_model = Model(inputs, decoder_model_output)

    return encoder_model, decoder_model, autoencoder_model


def plot_original_vs_generated(original, generated):
    num_images = 15
    sample = np.random.randint(0, len(original), num_images)

    def stack(data):
        images = data[sample]
        return np.vstack([np.hstack(images[:5]),
                          np.hstack(images[5:10]),
                          np.hstack(images[10:15])])

    def add_text(image, text, position):
        pt1 = position
        pt2 = (pt1[0] + 10 + (len(text) * 22),
               pt1[1] - 45)
        cv2.rectangle(image,
                      pt1,
                      pt2,
                      (255, 255, 255),
                      -1)
        cv2.putText(image, text,
                    position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.3,
                    color=(0, 0, 0),
                    thickness=4)

    original = stack(original)
    generated = stack(generated)

    mosaic = np.vstack([original,
                        generated])
    mosaic = cv2.resize(mosaic, (860, 860),
                        interpolation=cv2.INTER_AREA)
    mosaic = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR)

    add_text(mosaic, 'Original', (50, 100))
    add_text(mosaic, 'Generated', (50, 520))

    cv2.imshow('Mosaic', mosaic)
    cv2.waitKey(0)


(X_train, _), (X_test, _) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

train_noise = np.random.normal(loc=0.5, scale=0.5,
                               size=X_train.shape)
test_noise = np.random.normal(loc=0.5, scale=0.5,
                              size=X_test.shape)

X_train_noisy = np.clip(X_train + train_noise, 0, 1)
X_test_noisy = np.clip(X_test + test_noise, 0, 1)

_, _, autoencoder = build_autoencoder(encoding_size=128)
autoencoder.compile(optimizer='adam', loss='mse')

EPOCHS = 300
BATCH_SIZE = 1024
autoencoder.fit(X_train_noisy, X_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(X_test_noisy, X_test))

predictions = autoencoder.predict(X_test)

original_shape = (X_test_noisy.shape[0], 28, 28)
predictions = predictions.reshape(original_shape)
X_test_noisy = X_test_noisy.reshape(original_shape)

predictions = (predictions * 255.0).astype('uint8')
X_test_noisy = (X_test_noisy * 255.0).astype('uint8')

plot_original_vs_generated(X_test_noisy, predictions)
