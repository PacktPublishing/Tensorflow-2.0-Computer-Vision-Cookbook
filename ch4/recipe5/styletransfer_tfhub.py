import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_hub import load


def load_image(image_path):
    dimension = 512
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    longest_dimension = max(shape)
    scale = dimension / longest_dimension

    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    return image[tf.newaxis, :]


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        tensor = tensor[0]

    return tensor


def show_image(image):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    plt.show()


module_url = ('https://tfhub.dev/google/magenta/'
              'arbitrary-image-stylization-v1-256/2')
hub_module = load(module_url)

image = load_image('ch4/recipe5/bmw.jpg')
show_image(image)

style_image = load_image('ch4/recipe5/pollock.jpg')
show_image(style_image)

results = hub_module(tf.constant(image),
                     tf.constant(style_image))
stylized_image = tensor_to_image(results[0])
show_image(stylized_image)
