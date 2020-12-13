import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfdata
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.losses import \
    SparseCategoricalCrossentropy
from tensorflow.keras.models import *
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


class UNet(object):
    def __init__(self,
                 input_size=(256, 256, 3),
                 output_channels=3):
        self.pretrained_model = MobileNetV2(
            input_shape=input_size,
            include_top=False,
            weights='imagenet')

        self.target_layers = [
            'block_1_expand_relu',
            'block_3_expand_relu',
            'block_6_expand_relu',
            'block_13_expand_relu',
            'block_16_project'
        ]

        self.input_size = input_size
        self.output_channels = output_channels

        self.model = self._create_model()
        loss = SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=RMSprop(),
                           loss=loss,
                           metrics=['accuracy'])

    @staticmethod
    def _upsample(filters, size, dropout=False):
        init = tf.random_normal_initializer(0.0, 0.02)

        layers = Sequential()
        layers.add(Conv2DTranspose(filters=filters,
                                   kernel_size=size,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer=init,
                                   use_bias=False))

        layers.add(BatchNormalization())

        if dropout:
            layers.add(Dropout(rate=0.5))

        layers.add(ReLU())

        return layers

    def _create_model(self):
        layers = [self.pretrained_model.get_layer(l).output
                  for l in self.target_layers]
        down_stack = Model(inputs=self.pretrained_model.input,
                           outputs=layers)
        down_stack.trainable = False

        up_stack = []

        for filters in (512, 256, 128, 64):
            up_block = self._upsample(filters, 4)
            up_stack.append(up_block)

        inputs = Input(shape=self.input_size)
        x = inputs

        skip_layers = down_stack(x)
        x = skip_layers[-1]
        skip_layers = reversed(skip_layers[:-1])

        for up, skip_connection in zip(up_stack, skip_layers):
            x = up(x)
            x = Concatenate()([x, skip_connection])

        init = tf.random_normal_initializer(0.0, 0.02)
        output = Conv2DTranspose(
            filters=self.output_channels,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=init)(x)

        return Model(inputs, outputs=output)

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


dataset, info = tfdata.load('oxford_iiit_pet',
                            with_info=True)

TRAIN_SIZE = info.splits['train'].num_examples
VALIDATION_SIZE = info.splits['test'].num_examples
BATCH_SIZE = 64
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

unet = UNet()
unet.train(train_dataset,
           epochs=30,
           steps_per_epoch=STEPS_PER_EPOCH,
           validation_steps=VALIDATION_STEPS,
           validation_dataset=test_dataset)

unet.evaluate(test_dataset)
