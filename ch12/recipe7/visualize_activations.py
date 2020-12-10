import cv2
import imutils
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import *


class GradGAM(object):
    def __init__(self, model, class_index, layer_name=None):
        self.class_index = class_index

        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    layer_name = layer.name
                    break

        self.grad_model = self._create_grad_model(model,
                                                  layer_name)

    def _create_grad_model(self, model, layer_name):
        return Model(inputs=[model.inputs],
                     outputs=[
                         model.get_layer(layer_name).output,
                         model.output])

    def compute_heatmap(self, image, epsilon=1e-8):
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            conv_outputs, preds = self.grad_model(inputs)
            loss = preds[:, self.class_index]

        grads = tape.gradient(loss, conv_outputs)
        guided_grads = (tf.cast(conv_outputs > 0, 'float32') *
                        tf.cast(grads > 0, 'float32') *
                        grads)

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(
            tf.multiply(weights, conv_outputs),
            axis=-1)

        height, width = image.shape[1:3]
        heatmap = cv2.resize(cam.numpy(), (width, height))

        min = heatmap.min()
        max = heatmap.max()
        heatmap = (heatmap - min) / ((max - min) + epsilon)
        heatmap = (heatmap * 255.0).astype('uint8')

        return heatmap

    def overlay_heatmap(self,
                        heatmap,
                        image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image,
                                 alpha,
                                 heatmap,
                                 1 - alpha,
                                 0)

        return heatmap, output


model = ResNet50(weights='imagenet')

image = load_img('dog.jpg', target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

predictions = model.predict(image)
i = np.argmax(predictions[0])

cam = GradGAM(model, i)
heatmap = cam.compute_heatmap(image)

original_image = cv2.imread('dog.jpg')
heatmap = cv2.resize(heatmap, (original_image.shape[1],
                               original_image.shape[0]))
heatmap, output = cam.overlay_heatmap(heatmap, original_image,
                                      alpha=0.5)

decoded = imagenet_utils.decode_predictions(predictions)
_, label, probability = decoded[0][0]

cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, f'{label}: {probability * 100:.2f}%',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2)

output = np.hstack([original_image, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imwrite('output.jpg', output)
