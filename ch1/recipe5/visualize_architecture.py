from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

input_layer = Input(shape=(64, 64, 3), name='input_layer')
convolution_1 = Conv2D(kernel_size=(2, 2),
                       padding='same',
                       strides=(2, 2),
                       filters=32,
                       name='convolution_1')(input_layer)
activation_1 = LeakyReLU(name='activation_1')(convolution_1)
batch_normalization_1 = BatchNormalization(name='batch_normalization_1')(activation_1)
pooling_1 = MaxPooling2D(pool_size=(2, 2),
                         strides=(1, 1),
                         padding='same',
                         name='pooling_1')(batch_normalization_1)

convolution_2 = Conv2D(kernel_size=(2, 2),
                       padding='same',
                       strides=(2, 2),
                       filters=64,
                       name='convolution_2')(pooling_1)
activation_2 = LeakyReLU(name='activation_2')(convolution_2)
batch_normalization_2 = BatchNormalization(name='batch_normalization_2')(activation_2)
pooling_2 = MaxPooling2D(pool_size=(2, 2),
                         strides=(1, 1),
                         padding='same',
                         name='pooling_2')(batch_normalization_2)
dropout = Dropout(rate=0.5, name='dropout')(pooling_2)

flatten = Flatten(name='flatten')(dropout)
dense_1 = Dense(units=256, name='dense_1')(flatten)
activation_3 = LeakyReLU(name='activation_3')(dense_1)
dense_2 = Dense(units=128, name='dense_2')(activation_3)
activation_4 = LeakyReLU(name='activation_4')(dense_2)
dense_3 = Dense(units=3, name='dense_3')(activation_4)
output = Softmax(name='output')(dense_3)

model = Model(inputs=input_layer, outputs=output, name='my_model')

print(model.summary())
plot_model(model, show_shapes=True, show_layer_names=True, to_file='my_model.jpg')

model_diagram = Image.open('my_model.jpg')
model_diagram.show()