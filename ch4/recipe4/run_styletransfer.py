import matplotlib.pyplot as plt
import tensorflow as tf

from ch4.recipe3.styletransfer import StyleTransferrer

tf.config.experimental_run_functions_eagerly(True)


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


def show_image(image):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    plt.show()


content = load_image('ch4/recipe4/bmw.jpg')
show_image(content)

style = load_image('ch4/recipe4/pollock.jpg')
show_image(style)

# Default
stylized_image = StyleTransferrer().transfer(style, content)
show_image(stylized_image)

# More epochs
stylized_image = StyleTransferrer().transfer(style, content,
                                             epochs=100)
show_image(stylized_image)
