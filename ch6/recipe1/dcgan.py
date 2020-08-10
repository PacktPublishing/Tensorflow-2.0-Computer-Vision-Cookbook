import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DCGAN(object):
    def __init__(self):
        self.loss = BinaryCrossentropy(from_logits=True)
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.generator_opt = Adam(learning_rate=1e-4)
        self.discriminator_opt = Adam(learning_rate=1e-4)

    @staticmethod
    def create_generator(alpha=0.2):
        input = Input(shape=(100,))
        x = Dense(units=7 * 7 * 256, use_bias=False)(input)
        x = LeakyReLU(alpha=alpha)(x)
        x = BatchNormalization()(x)

        x = Reshape((7, 7, 256))(x)

        x = Conv2DTranspose(filters=128,
                            strides=(1, 1),
                            kernel_size=(5, 5),
                            padding='same',
                            use_bias=False)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=64,
                            strides=(2, 2),
                            kernel_size=(5, 5),
                            padding='same',
                            use_bias=False)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(filters=1,
                            strides=(2, 2),
                            kernel_size=(5, 5),
                            padding='same',
                            use_bias=False)(x)
        output = Activation('tanh')(x)

        return Model(input, output)

    @staticmethod
    def create_discriminator(alpha=0.2, dropout=0.3):
        input = Input(shape=(28, 28, 1))
        x = Conv2D(filters=64,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   padding='same')(input)
        x = LeakyReLU(alpha=alpha)(x)
        x = Dropout(rate=dropout)(x)

        x = Conv2D(filters=128,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   padding='same')(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Dropout(rate=dropout)(x)

        x = Flatten()(x)
        output = Dense(units=1)(x)

        return Model(input, output)

    def discriminator_loss(self, real, fake):
        real_loss = self.loss(tf.ones_like(real), real)
        fake_loss = self.loss(tf.zeros_like(fake), fake)

        return real_loss + fake_loss

    def generator_loss(self, fake):
        return self.loss(tf.ones_like(fake), fake)

    @tf.function
    def train_step(self, images, batch_size):
        noise = tf.random.normal((batch_size, noise_dimension))

        with tf.GradientTape() as gen_tape, \
                tf.GradientTape() as dis_tape:
            generated_images = self.generator(noise,
                                              training=True)

            real = self.discriminator(images, training=True)
            fake = self.discriminator(generated_images,
                                      training=True)

            gen_loss = self.generator_loss(fake)
            disc_loss = self.discriminator_loss(real, fake)

        generator_grad = gen_tape \
            .gradient(gen_loss,
                      self.generator.trainable_variables)
        discriminator_grad = dis_tape \
            .gradient(disc_loss,
                      self.discriminator.trainable_variables)

        opt_args = zip(generator_grad,
                       self.generator.trainable_variables)
        self.generator_opt.apply_gradients(opt_args)

        opt_args = zip(discriminator_grad,
                       self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(opt_args)

    def train(self, dataset, test_seed, epochs, batch_size):
        for epoch in tqdm(range(epochs)):
            for image_batch in dataset:
                self.train_step(image_batch, batch_size)

            if epoch == 0 or epoch % 10 == 0:
                generate_and_save_images(self.generator,
                                         epoch,
                                         test_seed)


def process_image(input):
    image = tf.cast(input['image'], tf.float32)
    image = (image - 127.5) / 127.5
    return image


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        image = predictions[i, :, :, 0] * 127.5 + 127.5
        image = tf.cast(image, tf.uint8)
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    plt.savefig(f'{epoch}.png')
    plt.show()


BUFFER_SIZE = 1000
BATCH_SIZE = 512

EPOCHS = 200
data = tfds.load('emnist', split='train')

train_dataset = (data
                 .map(process_image,
                      num_parallel_calls=AUTOTUNE)
                 .shuffle(BUFFER_SIZE)
                 .batch(BATCH_SIZE))

noise_dimension = 100
num_examples_to_generate = 16
seed_shape = (num_examples_to_generate, noise_dimension)
test_seed = tf.random.normal(seed_shape)

dcgan = DCGAN()
dcgan.train(train_dataset, test_seed, EPOCHS, BATCH_SIZE)
