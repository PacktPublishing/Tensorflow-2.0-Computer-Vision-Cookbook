import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE


def random_crop(image):
    return tf.image.random_crop(image, size=(256, 256, 3))


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1

    return image


def random_jitter(image):
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    image = tf.image.resize(image, (286, 286), method=method)
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image


def preprocess_training_image(image, _):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_test_image(image, _):
    image = normalize(image)
    return image


def generate_images(model, test_input, epoch):
    prediction = model(test_input)

    image = np.hstack([test_input[0], prediction[0]])
    image *= 0.5
    image += 0.5
    image *= 255.0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'{epoch + 1}.jpg', image)


class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        init = tf.random_normal_initializer(1.0, 0.02)
        self.scale = self.add_weight(name='scale',
                                     shape=input_shape[-1:],
                                     initializer=init,
                                     trainable=True)

        self.offset = self.add_weight(name='offset',
                                      shape=input_shape[-1:],
                                      initializer='zeros',
                                      trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x,
                                       axes=(1, 2),
                                       keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv

        return self.scale * normalized + self.offset


class CycleGAN(object):
    def __init__(self, output_channels=3, lambda_value=10):
        self.output_channels = output_channels
        self._lambda = lambda_value
        self.loss = BinaryCrossentropy(from_logits=True)

        self.gen_g = self.create_generator()
        self.gen_f = self.create_generator()

        self.dis_x = self.create_discriminator()
        self.dis_y = self.create_discriminator()

        self.gen_g_opt = Adam(learning_rate=2e-4, beta_1=0.5)
        self.gen_f_opt = Adam(learning_rate=2e-4, beta_1=0.5)

        self.dis_x_opt = Adam(learning_rate=2e-4, beta_1=0.5)
        self.dis_y_opt = Adam(learning_rate=2e-4, beta_1=0.5)

    @staticmethod
    def downsample(filters, size, norm=True):
        initializer = tf.random_normal_initializer(0.0, 0.02)

        layers = Sequential()
        layers.add(Conv2D(filters=filters,
                          kernel_size=size,
                          strides=2,
                          padding='same',
                          kernel_initializer=initializer,
                          use_bias=False))

        if norm:
            layers.add(InstanceNormalization())

        layers.add(LeakyReLU())

        return layers

    @staticmethod
    def upsample(filters, size, dropout=False):
        init = tf.random_normal_initializer(0.0, 0.02)

        layers = Sequential()
        layers.add(Conv2DTranspose(filters=filters,
                                   kernel_size=size,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer=init,
                                   use_bias=False))

        layers.add(InstanceNormalization())

        if dropout:
            layers.add(Dropout(rate=0.5))

        layers.add(ReLU())

        return layers

    def create_generator(self):
        down_stack = [
            self.downsample(64, 4, norm=False),
            self.downsample(128, 4),
            self.downsample(256, 4)]

        for _ in range(5):
            down_block = self.downsample(512, 4)
            down_stack.append(down_block)

        up_stack = []
        for _ in range(3):
            up_block = self.upsample(512, 4, dropout=True)
            up_stack.append(up_block)

        for filters in (512, 256, 128, 64):
            up_block = self.upsample(filters, 4)
            up_stack.append(up_block)

        inputs = Input(shape=(None, None, 3))
        x = inputs

        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = Concatenate()([x, skip])

        init = tf.random_normal_initializer(0.0, 0.02)
        output = Conv2DTranspose(
            filters=self.output_channels,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=init,
            activation='tanh')(x)

        return Model(inputs, outputs=output)

    def generator_loss(self, generated):
        return self.loss(tf.ones_like(generated), generated)

    def create_discriminator(self):
        input = Input(shape=(None, None, 3))
        x = input

        x = self.downsample(64, 4, False)(x)
        x = self.downsample(128, 4)(x)
        x = self.downsample(256, 4)(x)

        x = ZeroPadding2D()(x)

        init = tf.random_normal_initializer(0.0, 0.02)
        x = Conv2D(filters=512,
                   kernel_size=4,
                   strides=1,
                   kernel_initializer=init,
                   use_bias=False)(x)
        x = InstanceNormalization()(x)

        x = LeakyReLU()(x)
        x = ZeroPadding2D()(x)
        output = Conv2D(filters=1,
                        kernel_size=4,
                        strides=1,
                        kernel_initializer=init)(x)

        return Model(inputs=input, outputs=output)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss(tf.ones_like(real), real)
        generated_loss = self.loss(tf.zeros_like(generated),
                                   generated)

        total_discriminator_loss = real_loss + generated_loss
        return total_discriminator_loss * 0.5

    def calculate_cycle_loss(self, real_image, cycled_image):
        error = real_image - cycled_image
        loss1 = tf.reduce_mean(tf.abs(error))
        return self._lambda * loss1

    def identity_loss(self, real_image, same_image):
        error = real_image - same_image
        loss = tf.reduce_mean(tf.abs(error))
        return self._lambda * 0.5 * loss

    @tf.function
    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            # G translates X to Y
            # F translates Y to X.
            fake_y = self.gen_g(real_x, training=True)
            cycled_x = self.gen_f(fake_y, training=True)

            fake_x = self.gen_f(real_y, training=True)
            cycled_y = self.gen_g(fake_x, training=True)

            # same_x and same_y are used for identity loss
            same_x = self.gen_f(real_x, training=True)
            same_y = self.gen_g(real_y, training=True)

            dis_real_x = self.dis_x(real_x, training=True)
            dis_real_y = self.dis_y(real_y, training=True)
            dis_fake_x = self.dis_x(fake_x,training=True)
            dis_fake_y = self.dis_y(fake_y, training=True)

            # Compute the loss
            gen_g_loss = self.generator_loss(dis_fake_y)
            gen_f_loss = self.generator_loss(dis_fake_x)

            cycle_x_loss = \
                self.calculate_cycle_loss(real_x, cycled_x)
            cycle_y_loss = \
                self.calculate_cycle_loss(real_y, cycled_y)
            total_cycle_loss = cycle_x_loss + cycle_y_loss

            # Total generator loss = adversarial loss + cycle loss
            identity_y_loss = \
                self.identity_loss(real_y, same_y)
            total_generator_g_loss = (gen_g_loss +
                                      total_cycle_loss +
                                      identity_y_loss)

            identity_x_loss = \
                self.identity_loss(real_x, same_x)
            total_generator_f_loss = (gen_f_loss +
                                      total_cycle_loss +
                                      identity_x_loss)

            dis_x_loss = \
                self.discriminator_loss(dis_real_x, dis_fake_x)
            dis_y_loss = \
                self.discriminator_loss(dis_real_y, dis_fake_y)

        # Calculate the gradients for generator and discriminator.
        gen_g_grads = tape.gradient(
            total_generator_g_loss,
            self.gen_g.trainable_variables)
        gen_f_grads = tape.gradient(
            total_generator_f_loss,
            self.gen_f.trainable_variables)

        dis_x_grads = tape.gradient(
            dis_x_loss,
            self.dis_x.trainable_variables)
        dis_y_grads = tape.gradient(
            dis_y_loss,
            self.dis_y.trainable_variables)

        # Apply the gradients to the optimizer
        gen_g_opt_params = zip(gen_g_grads,
                               self.gen_g.trainable_variables)
        self.gen_g_opt.apply_gradients(gen_g_opt_params)

        gen_f_opt_params = zip(gen_f_grads,
                               self.gen_f.trainable_variables)
        self.gen_f_opt.apply_gradients(gen_f_opt_params)

        dis_x_opt_params = zip(dis_x_grads,
                               self.dis_x.trainable_variables)
        self.dis_x_opt.apply_gradients(dis_x_opt_params)

        dis_y_opt_params = zip(dis_y_grads,
                               self.dis_y.trainable_variables)
        self.dis_y_opt.apply_gradients(dis_y_opt_params)

    def fit(self, train, epochs, test):
        for epoch in tqdm(range(epochs)):
            for image_x, image_y in train:
                self.train_step(image_x, image_y)

            test_image = next(iter(test))
            generate_images(self.gen_g, test_image, epoch)


dataset, _ = tfds.load('cycle_gan/summer2winter_yosemite',
                       with_info=True,
                       as_supervised=True)

train_summer = dataset['trainA']
train_winter = dataset['trainB']

test_summer = dataset['testA']
test_winter = dataset['testB']

BUFFER_SIZE = 400
BATCH_SIZE = 1

train_summer = (train_summer
                .map(preprocess_training_image,
                     num_parallel_calls=AUTOTUNE)
                .cache()
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE))
train_winter = (train_winter
                .map(preprocess_training_image,
                     num_parallel_calls=AUTOTUNE)
                .cache()
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE))

test_summer = (test_summer
               .map(preprocess_test_image,
                    num_parallel_calls=AUTOTUNE)
               .cache()
               .shuffle(BUFFER_SIZE)
               .batch(BATCH_SIZE))
test_winter = (test_winter
               .map(preprocess_test_image,
                    num_parallel_calls=AUTOTUNE)
               .cache()
               .shuffle(BUFFER_SIZE)
               .batch(BATCH_SIZE))

cycle_gan = CycleGAN()
train_ds = tf.data.Dataset.zip((train_summer, train_winter))
cycle_gan.fit(train=train_ds,
              epochs=40,
              test=test_summer)
