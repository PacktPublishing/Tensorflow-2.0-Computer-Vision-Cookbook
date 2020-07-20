import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.datasets import fashion_mnist as fmnist
from tensorflow.keras.layers import *

SEED = 84
np.random.seed(SEED)


def build_autoencoder(input_shape=(28, 28, 1),
                      encoding_size=96,
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
    encoder_output = Dense(encoding_size)(encoder)

    encoder_model = Model(inputs, encoder_output)

    # Build decoder
    decoder_input = Input(shape=(encoding_size,))
    target_shape = tuple(encoder_output_shape[1:])
    decoder = Dense(np.prod(target_shape))(decoder_input)
    decoder = Reshape(target_shape)(decoder)

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


def create_anomalous_dataset(features,
                             labels,
                             regular_label,
                             anomaly_label,
                             corruption_proportion=0.01):
    regular_data_idx = np.where(labels == regular_label)[0]
    anomalous_data_idx = np.where(labels == anomaly_label)[0]

    np.random.shuffle(regular_data_idx)
    np.random.shuffle(anomalous_data_idx)

    num_anomalies = int(len(regular_data_idx) *
                        corruption_proportion)
    anomalous_data_idx = anomalous_data_idx[:num_anomalies]

    data = np.vstack([features[regular_data_idx],
                      features[anomalous_data_idx]])
    np.random.shuffle(data)

    return data


(X_train, y_train), (X_test, y_test) = fmnist.load_data()
X = np.vstack([X_train, X_test])
y = np.hstack([y_train, y_test])

REGULAR_LABEL = 5  # Sandal
ANOMALY_LABEL = 0  # T-shirt/top

data = create_anomalous_dataset(X, y,
                                REGULAR_LABEL,
                                ANOMALY_LABEL)

data = np.expand_dims(data, axis=-1)
data = data.astype('float32') / 255.0

X_train, X_test = train_test_split(data,
                                   train_size=0.8,
                                   random_state=SEED)

_, _, autoencoder = build_autoencoder(encoding_size=256)
autoencoder.compile(optimizer='adam', loss='mse')

EPOCHS = 300
BATCH_SIZE = 1024
autoencoder.fit(X_train, X_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, X_test))

decoded = autoencoder.predict(data)
mses = []
for original, generated in zip(data, decoded):
    mse = np.mean((original - generated) ** 2)
    mses.append(mse)

threshold = np.quantile(mses, 0.999)
outlier_idx = np.where(np.array(mses) >= threshold)[0]
print(f'Number of outliers: {len(outlier_idx)}')

decoded = (decoded * 255.0).astype('uint8')
data = (data * 255.0).astype('uint8')

for i in outlier_idx:
    image = np.hstack([data[i].reshape(28, 28),
                       decoded[i].reshape(28, 28)])
    cv2.imwrite(f'{i}.jpg', image)
