from autokeras import *
from tensorflow.keras.datasets import fashion_mnist as fm
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

(X_train, y_train), (X_test, y_test) = fm.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

EPOCHS = 10

classifier = ImageClassifier(seed=9,
                             max_trials=20,
                             optimizer='adam')
classifier.fit(X_train, y_train, epochs=EPOCHS, verbose=2)
model = classifier.export_model()
model.save('model.h5')

model = load_model('model.h5',
                   custom_objects=CUSTOM_OBJECTS)
print(classifier.evaluate(X_test, y_test))

print(model.summary())
plot_model(model,
           show_shapes=True,
           show_layer_names=True,
           to_file='model.png')
