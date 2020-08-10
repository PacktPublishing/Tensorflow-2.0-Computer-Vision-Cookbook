import cv2
import tensorflow as tf
from tensorflow.keras.applications.nasnet import *
from tensorflow.keras.losses import CategoricalCrossentropy


def preprocess(image, target_shape):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    image = preprocess_input(image)
    image = image[None, :, :, :]
    return image


def get_imagenet_label(probabilities):
    return decode_predictions(probabilities, top=1)[0][0]


def save_image(image, model, description):
    prediction = model.predict(image)
    _, label, conf = get_imagenet_label(prediction)
    image = image.numpy()[0] * 0.5 + 0.5
    image = (image * 255).astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    conf *= 100
    img_name = f'{description}, {label} ({conf:.2f}%).jpg'
    cv2.imwrite(img_name, image)


def generate_adv_pattern(model,
                         input_image,
                         input_label,
                         loss_function):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_function(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_gradient = tf.sign(gradient)

    return signed_gradient


pretrained_model = NASNetMobile(include_top=True,
                                weights='imagenet')
pretrained_model.trainable = False

image = tf.io.read_file('dog.jpg')
image = tf.image.decode_jpeg(image)

image = preprocess(image, pretrained_model.input.shape[1:-1])
image_probabilities = pretrained_model.predict(image)

cce_loss = CategoricalCrossentropy()

pug_index = 254
label = tf.one_hot(pug_index, image_probabilities.shape[-1])
label = tf.reshape(label, (1, image_probabilities.shape[-1]))
disturbances = generate_adv_pattern(pretrained_model,
                                    image,
                                    label,
                                    cce_loss)

for epsilon in [0, 0.005, 0.01, 0.1, 0.15, 0.2]:
    corrupted_image = image + epsilon * disturbances
    corrupted_image = tf.clip_by_value(corrupted_image, -1, 1)
    save_image(corrupted_image,
               pretrained_model,
               f'Epsilon = {epsilon:.3f}')
