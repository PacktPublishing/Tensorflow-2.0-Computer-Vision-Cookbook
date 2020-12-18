from autokeras import *
from tensorflow.keras.datasets import fashion_mnist as fm
from tensorflow.keras.utils import *

(X_train, y_train), (X_test, y_test) = fm.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


def create_automodel(max_trials=10):
    input = ImageInput()
    x = Normalization()(input)
    x = ImageAugmentation(horizontal_flip=False,
                          vertical_flip=False)(x)

    left = ConvBlock()(x)
    right = XceptionBlock(pretrained=True)(x)

    x = Merge()([left, right])
    x = SpatialReduction(reduction_type='flatten')(x)
    x = DenseBlock()(x)

    output = ClassificationHead()(x)

    return AutoModel(inputs=input,
                     outputs=output,
                     overwrite=True,
                     max_trials=max_trials)


EPOCHS = 10

model = create_automodel()
model.fit(X_train, y_train, epochs=EPOCHS)

model = model.export_model()
print(model.evaluate(X_test, to_categorical(y_test)))

plot_model(model,
           show_shapes=True,
           show_layer_names=True,
           to_file='automodel.png')
