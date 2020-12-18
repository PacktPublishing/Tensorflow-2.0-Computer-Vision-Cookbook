from autokeras import ImageClassifier
from tensorflow.keras.datasets import fashion_mnist as fm

(X_train, y_train), (X_test, y_test) = fm.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

EPOCHS = 10

classifier = ImageClassifier(seed=9, max_trials=10)
classifier.fit(X_train, y_train, epochs=EPOCHS, verbose=2)
print(classifier.evaluate(X_test, y_test))
