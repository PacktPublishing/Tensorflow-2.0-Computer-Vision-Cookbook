import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam

tf.config.experimental_run_functions_eagerly(True)


class VAE(object):
    def __init__(self,
                 original_dimension=784,
                 encoding_dimension=512,
                 latent_dimension=2):
        self.original_dimension = original_dimension
        self.encoding_dimension = encoding_dimension
        self.latent_dimension = latent_dimension

        self.z_log_var = None
        self.z_mean = None

        self.inputs = None
        self.outputs = None

        self.encoder = None
        self.decoder = None
        self.vae = None

    def build_vae(self):
        # Build encoder
        self.inputs = Input(shape=(self.original_dimension,))
        x = Dense(self.encoding_dimension)(self.inputs)
        x = ReLU()(x)
        self.z_mean = Dense(self.latent_dimension)(x)
        self.z_log_var = Dense(self.latent_dimension)(x)

        z = Lambda(sampling)([self.z_mean, self.z_log_var])

        self.encoder = Model(self.inputs,
                             [self.z_mean, self.z_log_var, z])

        # Build decoder
        latent_inputs = Input(shape=(self.latent_dimension,))
        x = Dense(self.encoding_dimension)(latent_inputs)
        x = ReLU()(x)
        self.outputs = Dense(self.original_dimension)(x)
        self.outputs = Activation('sigmoid')(self.outputs)
        self.decoder = Model(latent_inputs, self.outputs)

        # Build end-to-end VAE.
        self.outputs = self.encoder(self.inputs)[2]
        self.outputs = self.decoder(self.outputs)
        self.vae = Model(self.inputs, self.outputs)

    @tf.function
    def train(self,
              X_train,
              X_test,
              epochs=50,
              batch_size=64):
        reconstruction_loss = mse(self.inputs, self.outputs)
        reconstruction_loss *= self.original_dimension

        kl_loss = (1 + self.z_log_var -
                   K.square(self.z_mean) -
                   K.exp(self.z_log_var))
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(reconstruction_loss + kl_loss)

        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=Adam(lr=1e-3))
        self.vae.fit(X_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(X_test, None))

        return self.encoder, self.decoder, self.vae


def sampling(arguments):
    z_mean, z_log_var = arguments
    batch = K.shape(z_mean)[0]
    dimension = K.int_shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch, dimension))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def generate_and_plot(decoder, grid_size=5):
    cell_size = 28

    figure_shape = (grid_size * cell_size,
                    grid_size * cell_size)
    figure = np.zeros(figure_shape)
    grid_x = np.linspace(-4, 4, grid_size)
    grid_y = np.linspace(-4, 4, grid_size)[::-1]

    for i, z_log_var in enumerate(grid_y):
        for j, z_mean in enumerate(grid_x):
            z_sample = np.array([[z_mean, z_log_var]])
            generated = decoder.predict(z_sample)[0]

            # Reshape as image.
            fashion_item = generated.reshape(cell_size,
                                             cell_size)

            # Assign to the corresponding cell in the grid.
            y_slice = slice(i * cell_size,
                            (i + 1) * cell_size)
            x_slice = slice(j * cell_size,
                            (j + 1) * cell_size)
            figure[y_slice, x_slice] = fashion_item

    plt.figure(figsize=(10, 10))
    start = cell_size // 2
    end = (grid_size - 2) * cell_size + start + 1
    pixel_range = np.arange(start, end, cell_size)

    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)

    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel('z_mean')
    plt.ylabel('z_log_var')
    plt.imshow(figure)
    plt.show()


(X_train, _), (X_test, _) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

vae = VAE(original_dimension=784,
          encoding_dimension=512,
          latent_dimension=2)
vae.build_vae()

_, decoder_model, vae_model = vae.train(X_train, X_test,
                                        epochs=100)
generate_and_plot(decoder_model, grid_size=7)
