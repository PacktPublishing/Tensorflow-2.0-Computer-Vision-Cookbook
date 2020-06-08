import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.utils import get_file

classifier_url = ('https://tfhub.dev/google/imagenet/'
                  'resnet_v2_152/classification/4')

model = Sequential([
    hub.KerasLayer(classifier_url, input_shape=(224, 224, 3))
])

image = load_img('beetle.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)

predictions = model.predict(image)

predicted_index = np.argmax(predictions[0], axis=-1)

file_name = 'ImageNetLabels.txt'
file_url = ('https://storage.googleapis.com/'
            'download.tensorflow.org/data/ImageNetLabels.txt')
labels_path = get_file(file_name, file_url)

with open(labels_path) as f:
    imagenet_labels = np.array(f.read().splitlines())

predicted_class = imagenet_labels[predicted_index]
print(predicted_class)

plt.figure()
plt.title(f'Label: {predicted_class}.')
original = load_img('beetle.jpg')
original = img_to_array(original)
plt.imshow(original / 255.0)
plt.show()
