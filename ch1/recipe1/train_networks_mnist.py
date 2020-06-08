from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import *

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28 * 28 * 1))
X_test = X_test.reshape((X_test.shape[0], 28 * 28 * 1))

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

sequential_model = Sequential()
sequential_model.add(Dense(256, input_shape=(28 * 28 * 1,), activation='sigmoid'))
sequential_model.add(Dense(128, activation='sigmoid'))
sequential_model.add(Dense(10, activation='softmax'))

layers = [Dense(256, input_shape=(28 * 28 * 1,), activation='sigmoid'),
          Dense(128, activation='sigmoid'),
          Dense(10, activation='softmax')]
sequential_model2 = Sequential(layers)

input_layer = Input(shape=(28 * 28 * 1,))
dense_1 = Dense(256, activation='sigmoid')(input_layer)
dense_2 = Dense(128, activation='sigmoid')(dense_1)
predictions = Dense(10, activation='softmax')(dense_2)
functional_model = Model(inputs=input_layer, outputs=predictions)


class ClassModel(Model):
    def __init__(self):
        super(ClassModel, self).__init__()

        self.dense_1 = Dense(256, activation='sigmoid')
        self.dense_2 = Dense(256, activation='sigmoid')
        self.predictions = Dense(10, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)

        return self.predictions(x)


class_model = ClassModel()

models = {
    'sequential_model': sequential_model,
    'sequential_model_list': sequential_model2,
    'functional_model': functional_model,
    'class_model': class_model
}

for name, model in models.items():
    print(f'Compiling model: {name}')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(f'Training model: {name}')
    model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid),
              epochs=50,
              batch_size=256,
              verbose=0)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Testing model: {name}. \nAccuracy: {accuracy}')
    print('---')
