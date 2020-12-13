import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from tensorflow.keras.layers import *
from tensorflow.keras.losses import \
    SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

AUTOTUNE = tf.data.experimental.AUTOTUNE


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask


@tf.function
def load_image(dataset_element, train=True):
    input_image = tf.image.resize(dataset_element['image'],
                                  (256, 256))
    input_mask = tf.image.resize(
        dataset_element['segmentation_mask'], (256, 256))

    if train and np.random.uniform() > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image,
                                        input_mask)

    return input_image, input_mask


class FCN(object):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 output_channels=3):
        self.input_shape = input_shape
        self.output_channels = output_channels

        self.vgg_weights_path = str(pathlib.Path.home() /
                                    '.keras' / 'models' /
                                    'vgg16_weights_tf_dim_'
                                    'ordering_tf_kernels.h5')

        self.model = self._create_model()

        loss = SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=RMSprop(),
                           loss=loss,
                           metrics=['accuracy'])
    def _create_model(self):
        input = Input(shape=self.input_shape)

        x = Conv2D(filters=64,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block1_conv1')(input)
        x = Conv2D(filters=64,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block1_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2),
                         strides=2,
                         name='block1_pool')(x)

        x = Conv2D(filters=128,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block2_conv1')(x)
        x = Conv2D(filters=128,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block2_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2),
                         strides=2,
                         name='block2_pool')(x)

        x = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block3_conv1')(x)
        x = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block3_conv2')(x)
        x = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block3_conv3')(x)
        x = MaxPooling2D(pool_size=(2, 2),
                         strides=2,
                         name='block3_pool')(x)
        block3_pool = x

        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block4_conv1')(x)
        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block4_conv2')(x)
        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block4_conv3')(x)
        block4_pool = MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   name='block4_pool')(x)

        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block5_conv1')(block4_pool)
        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block5_conv2')(x)
        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='block5_conv3')(x)
        block5_pool = MaxPooling2D(pool_size=(2, 2),
                                   strides=2,
                                   name='block5_pool')(x)

        model = Model(input, block5_pool)
        model.load_weights(self.vgg_weights_path,
                           by_name=True)

        output = Conv2D(filters=self.output_channels,
                        kernel_size=(7, 7),
                        activation='relu',
                        padding='same',
                        name='conv6')(block5_pool)

        conv6_4 = Conv2DTranspose(
            filters=self.output_channels,
            kernel_size=(4, 4),
            strides=4,
            use_bias=False)(output)
        pool4_n = Conv2D(filters=self.output_channels,
                         kernel_size=(1, 1),
                         activation='relu',
                         padding='same',
                         name='pool4_n')(block4_pool)
        pool4_n_2 = Conv2DTranspose(
            filters=self.output_channels,
            kernel_size=(2, 2),
            strides=2,
            use_bias=False)(pool4_n)

        pool3_n = Conv2D(filters=self.output_channels,
                         kernel_size=(1, 1),
                         activation='relu',
                         padding='same',
                         name='pool3_n')(block3_pool)

        output = Add(name='add')([pool4_n_2,
                                  pool3_n,
                                  conv6_4])
        output = Conv2DTranspose(filters=self.output_channels,
                                 kernel_size=(8, 8),
                                 strides=8,
                                 use_bias=False)(output)
        output = Softmax()(output)

        return Model(input, output)

    @staticmethod
    def _plot_model_history(model_history, metric, ylim=None):
        plt.style.use('seaborn-darkgrid')
        plotter = tfdocs.plots.HistoryPlotter()
        plotter.plot({'Model': model_history}, metric=metric)

        plt.title(f'{metric.upper()}')
        if ylim is None:
            plt.ylim([0, 1])
        else:
            plt.ylim(ylim)

        plt.savefig(f'{metric}.png')
        plt.close()

    def train(self, train_dataset, epochs, steps_per_epoch,
              validation_dataset, validation_steps):

        hist = \
            self.model.fit(train_dataset,
                           epochs=epochs,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=validation_steps,
                           validation_data=validation_dataset)

        self._plot_model_history(hist, 'loss', [0., 2.0])
        self._plot_model_history(hist, 'accuracy')

    @staticmethod
    def _process_mask(mask):
        mask = (mask.numpy() * 127.5).astype('uint8')
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        return mask

    def _save_image_and_masks(self, image,
                              ground_truth_mask,
                              prediction_mask,
                              image_id):
        image = (image.numpy() * 255.0).astype('uint8')
        gt_mask = self._process_mask(ground_truth_mask)
        pred_mask = self._process_mask(prediction_mask)

        mosaic = np.hstack([image, gt_mask, pred_mask])
        mosaic = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'mosaic_{image_id}.jpg', mosaic)

    @staticmethod
    def _create_mask(prediction_mask):
        prediction_mask = tf.argmax(prediction_mask, axis=-1)
        prediction_mask = prediction_mask[..., tf.newaxis]

        return prediction_mask[0]

    def _save_predictions(self, dataset, sample_size=1):
        for id, (image, mask) in \
                enumerate(dataset.take(sample_size), start=1):
            pred_mask = self.model.predict(image)
            pred_mask = self._create_mask(pred_mask)

            image = image[0]
            ground_truth_mask = mask[0]

            self._save_image_and_masks(image,
                                       ground_truth_mask,
                                       pred_mask,
                                       image_id=id)

    def evaluate(self, test_dataset, sample_size=5):
        result = self.model.evaluate(test_dataset)
        print(f'Accuracy: {result[1] * 100:.2f}%')

        self._save_predictions(test_dataset, sample_size)


dataset, info = tfds.load('oxford_iiit_pet', with_info=True)

TRAIN_SIZE = info.splits['train'].num_examples
VALIDATION_SIZE = info.splits['test'].num_examples
BATCH_SIZE = 32
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE

VALIDATION_SUBSPLITS = 5
VALIDATION_STEPS = VALIDATION_SIZE // BATCH_SIZE
VALIDATION_STEPS //= VALIDATION_SUBSPLITS

BUFFER_SIZE = 1000
train_dataset = (dataset['train']
                 .map(load_image, num_parallel_calls=AUTOTUNE)
                 .cache()
                 .shuffle(BUFFER_SIZE)
                 .batch(BATCH_SIZE)
                 .repeat()
                 .prefetch(buffer_size=AUTOTUNE))
test_dataset = (dataset['test']
                .map(lambda d: load_image(d, train=False),
                     num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE))

fcn = FCN(output_channels=3)
fcn.train(train_dataset,
          epochs=120,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_steps=VALIDATION_STEPS,
          validation_dataset=test_dataset)

fcn.evaluate(test_dataset)
