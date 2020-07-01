import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import *

from ch4.recipe1.deepdream import DeepDreamer


def load_image(image_path):
    image = load_img(image_path)
    image = img_to_array(image)

    return image


def show_image(image):
    plt.imshow(image)
    plt.show()


original_image = load_image('ch4/recipe2/road.jpg')
show_image(original_image / 255.0)

dreamy_image = DeepDreamer().dream(original_image)
show_image(dreamy_image)

dreamy_image = (DeepDreamer(layers=['mixed2',
                                    'mixed5',
                                    'mixed7'])
                .dream(original_image))
show_image(dreamy_image)

dreamy_image = (DeepDreamer(octave_power_factors=[-3, -1,
                                                  0, 3])
                .dream(original_image))
show_image(dreamy_image)
