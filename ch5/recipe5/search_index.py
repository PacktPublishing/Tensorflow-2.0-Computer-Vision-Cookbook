import cv2
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import *


def build_autoencoder(input_shape=(28, 28, 1),
                      encoding_size=32,
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
    encoder_output = Dense(units=encoding_size,
                           name='encoder_output')(encoder)
    # Build decoder
    target_shape = tuple(encoder_output_shape[1:])
    decoder = Dense(np.prod(target_shape))(encoder_output)
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
    outputs = Activation(activation='sigmoid',
                         name='decoder_output')(decoder)
    autoencoder_model = Model(inputs, outputs)

    return autoencoder_model


def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


def search(query_vector, search_index, max_results=16):
    vectors = search_index['features']
    results = []

    for i in range(len(vectors)):
        distance = euclidean_dist(query_vector, vectors[i])
        results.append((distance, search_index['images'][i]))

    results = sorted(results,
                     key=lambda p: p[0])[:max_results]
    return results


(X_train, _), (X_test, _) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

EPOCHS = 10
BATCH_SIZE = 512
autoencoder.fit(X_train, X_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(X_test, X_test))

fe_input = autoencoder.input
fe_output = autoencoder.get_layer('encoder_output').output
feature_extractor = Model(inputs=fe_input, outputs=fe_output)

train_vectors = feature_extractor.predict(X_train)

X_train = (X_train * 255.0).astype('uint8')
X_train = X_train.reshape((X_train.shape[0], 28, 28))
search_index = {
    'features': train_vectors,
    'images': X_train
}

test_vectors = feature_extractor.predict(X_test)

X_test = (X_test * 255.0).astype('uint8')
X_test = X_test.reshape((X_test.shape[0], 28, 28))

sample_indices = np.random.randint(0, X_test.shape[0], 16)
sample_images = X_test[sample_indices]
sample_queries = test_vectors[sample_indices]

for i, (vector, image) in \
        enumerate(zip(sample_queries, sample_images)):
    results = search(vector, search_index)
    results = [r[1] for r in results]
    query_image = cv2.resize(image, (28 * 4, 28 * 4),
                             interpolation=cv2.INTER_AREA)

    results_mosaic = np.vstack([np.hstack(results[0:4]),
                                np.hstack(results[4:8]),
                                np.hstack(results[8:12]),
                                np.hstack(results[12:16])])
    result_image = np.hstack([query_image, results_mosaic])
    cv2.imwrite(f'{i}.jpg', result_image)
